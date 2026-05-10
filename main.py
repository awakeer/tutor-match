"""
家教订单匹配系统 - MVP
单文件实现，方便部署和理解。
"""
import os
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import bcrypt
from openai import OpenAI
from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()

# ---------- 配置 ----------
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "tutor_match.db"
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "changeme")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
SESSION_SECRET = os.getenv("SESSION_SECRET", secrets.token_hex(32))

# 标准化词表 - 用于筛选下拉框和数据归一
LEVELS = ["小学", "初中", "高中"]
SUBJECTS = ["语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "政治", "其他"]
DISTRICTS = ["上城", "拱墅", "西湖", "滨江", "萧山", "余杭", "临平", "钱塘", "富阳", "临安", "桐庐", "淳安", "建德", "其他"]
TIME_SLOTS = ["工作日白天", "工作日晚上", "周末白天", "周末晚上"]

# ---------- 数据库 ----------
def init_db():
    """初始化数据库结构"""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS tutors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        real_name TEXT NOT NULL,
        phone TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        tutor_id INTEGER NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (tutor_id) REFERENCES tutors(id)
    );

    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        level TEXT,
        subject TEXT,
        grade TEXT,
        district TEXT,
        price_per_hour INTEGER,
        time_slots TEXT,
        requirements TEXT,
        parent_contact TEXT,
        raw_text TEXT,
        status TEXT DEFAULT 'open',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER NOT NULL,
        tutor_id INTEGER NOT NULL,
        message TEXT,
        handled INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (order_id) REFERENCES orders(id),
        FOREIGN KEY (tutor_id) REFERENCES tutors(id),
        UNIQUE(order_id, tutor_id)
    );

    CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
    CREATE INDEX IF NOT EXISTS idx_orders_level ON orders(level);
    CREATE INDEX IF NOT EXISTS idx_orders_district ON orders(district);
    """)
    conn.commit()
    conn.close()


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------- LLM 拆单 ----------
PARSE_SYSTEM_PROMPT = """你是一个家教订单解析助手。用户会贴给你一段微信群的聊天记录，里面混杂着家教需求、闲聊、广告等内容。

你的任务是从中提取出所有"家长发布的家教需求"，输出为严格的 JSON 数组。

每条单子的结构：
{
  "level": "小学|初中|高中",
  "subject": "语文|数学|英语|物理|化学|生物|历史|地理|政治|其他",
  "grade": "具体年级，如'高二''初三''小学五年级'",
  "district": "区域，提取到区级别，如'海淀''朝阳'，没提到填null",
  "price_per_hour": 时薪整数（元/小时），没提到填null。如果给的是总价或日价要尝试换算
  "time_slots": ["工作日白天"|"工作日晚上"|"周末白天"|"周末晚上"]，从原文判断，可多选
  "requirements": "其他要求的简短总结，如'要求女老师、有经验'，没特别要求填空字符串",
  "parent_contact": "家长联系方式（微信号或手机号），原样保留，没有填null",
  "raw_text": "这条单的原始文本片段"
}

重要规则：
1. 严格输出 JSON 数组，不要解释、不要markdown代码块、不要其他文字
2. 拆单粒度：一条家教需求一条记录。如果一个家长同时发了"高一数学+物理"，拆成两条
3. 只提取家长发布的需求。老师的自我推荐、群主的通知、闲聊广告全部跳过
4. 字段不确定就填null，不要瞎猜
5. 如果整段文本里没有任何家教需求，输出空数组 []
"""


def parse_orders_from_text(raw_text: str) -> list[dict]:
    """调用 DeepSeek API 把群聊文本拆成结构化单子"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=4096,
        messages=[
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ],
    )

    text = response.choices[0].message.content.strip()
    # 兜底：如果模型还是用了代码块就剥掉
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM返回不是合法JSON: {e}\n原始返回: {text[:500]}")


# ---------- 认证 ----------
def hash_password(pwd: str) -> str:
    return bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()


def verify_password(pwd: str, hashed: str) -> bool:
    return bcrypt.checkpw(pwd.encode(), hashed.encode())


def get_current_tutor(session: Optional[str] = Cookie(None)):
    """从session cookie拿当前老师"""
    if not session:
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT t.* FROM sessions s JOIN tutors t ON s.tutor_id = t.id WHERE s.token = ?",
            (session,)
        ).fetchone()
        return dict(row) if row else None


def require_tutor(tutor=Depends(get_current_tutor)):
    if not tutor:
        raise HTTPException(status_code=302, headers={"Location": "/login"})
    return tutor


def require_admin(request: Request):
    """HTTP Basic Auth 保护管理后台"""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Basic "):
        raise HTTPException(401, headers={"WWW-Authenticate": "Basic"})
    import base64
    try:
        decoded = base64.b64decode(auth[6:]).decode()
        u, p = decoded.split(":", 1)
    except Exception:
        raise HTTPException(401, headers={"WWW-Authenticate": "Basic"})
    if u != ADMIN_USER or p != ADMIN_PASS:
        raise HTTPException(401, headers={"WWW-Authenticate": "Basic"})
    return True


# ---------- FastAPI 应用 ----------
app = FastAPI(title="家教订单匹配系统")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

init_db()


# ---------- 公开页面（老师用） ----------
@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    level: str = "",
    subject: str = "",
    district: str = "",
    time_slot: str = "",
    min_price: int = 0,
    sort: str = "newest",
    tutor=Depends(get_current_tutor),
):
    """老师筛选首页"""
    sql = "SELECT * FROM orders WHERE status = 'open'"
    params = []
    
    if level:
        sql += " AND level = ?"
        params.append(level)
    if subject:
        sql += " AND subject = ?"
        params.append(subject)
    if district:
        sql += " AND district = ?"
        params.append(district)
    if time_slot:
        sql += " AND time_slots LIKE ?"
        params.append(f"%{time_slot}%")
    if min_price > 0:
        sql += " AND price_per_hour >= ?"
        params.append(min_price)
    
    if sort == "price":
        sql += " ORDER BY price_per_hour DESC NULLS LAST, created_at DESC"
    else:
        sql += " ORDER BY created_at DESC"
    
    with get_db() as conn:
        orders = [dict(r) for r in conn.execute(sql, params).fetchall()]
        # 老师已经申请过的单子
        applied_ids = set()
        if tutor:
            applied = conn.execute(
                "SELECT order_id FROM applications WHERE tutor_id = ?",
                (tutor["id"],)
            ).fetchall()
            applied_ids = {r["order_id"] for r in applied}
    
    # 解析 time_slots JSON 字符串便于模板使用
    for o in orders:
        try:
            o["time_slots_list"] = json.loads(o["time_slots"]) if o["time_slots"] else []
        except Exception:
            o["time_slots_list"] = []
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "orders": orders,
        "tutor": tutor,
        "applied_ids": applied_ids,
        "filters": {
            "level": level, "subject": subject, "district": district,
            "time_slot": time_slot, "min_price": min_price, "sort": sort,
        },
        "LEVELS": LEVELS,
        "SUBJECTS": SUBJECTS,
        "DISTRICTS": DISTRICTS,
        "TIME_SLOTS": TIME_SLOTS,
    })


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "error": None})


@app.post("/register")
def register(
    username: str = Form(...),
    password: str = Form(...),
    real_name: str = Form(...),
    phone: str = Form(...),
):
    if len(password) < 6:
        return templates.TemplateResponse(
            "register.html",
            {"request": Request, "error": "密码至少6位"},
            status_code=400,
        )
    with get_db() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO tutors (username, password_hash, real_name, phone) VALUES (?, ?, ?, ?)",
                (username, hash_password(password), real_name, phone),
            )
            tutor_id = cur.lastrowid
            token = secrets.token_urlsafe(32)
            conn.execute("INSERT INTO sessions (token, tutor_id) VALUES (?, ?)", (token, tutor_id))
        except sqlite3.IntegrityError:
            return HTMLResponse(
                "<p>用户名已存在 <a href='/register'>返回</a></p>",
                status_code=400,
            )
    resp = RedirectResponse("/", status_code=302)
    resp.set_cookie("session", token, httponly=True, max_age=30 * 86400)
    return resp


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM tutors WHERE username = ?", (username,)).fetchone()
        if not row or not verify_password(password, row["password_hash"]):
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "用户名或密码错误"},
                status_code=400,
            )
        token = secrets.token_urlsafe(32)
        conn.execute("INSERT INTO sessions (token, tutor_id) VALUES (?, ?)", (token, row["id"]))
    resp = RedirectResponse("/", status_code=302)
    resp.set_cookie("session", token, httponly=True, max_age=30 * 86400)
    return resp


@app.post("/logout")
def logout(session: Optional[str] = Cookie(None)):
    if session:
        with get_db() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (session,))
    resp = RedirectResponse("/", status_code=302)
    resp.delete_cookie("session")
    return resp


@app.post("/apply/{order_id}")
def apply_order(order_id: int, message: str = Form(""), tutor=Depends(require_tutor)):
    """老师申请接单"""
    with get_db() as conn:
        order = conn.execute("SELECT * FROM orders WHERE id = ? AND status = 'open'", (order_id,)).fetchone()
        if not order:
            raise HTTPException(404, "单子不存在或已关闭")
        try:
            conn.execute(
                "INSERT INTO applications (order_id, tutor_id, message) VALUES (?, ?, ?)",
                (order_id, tutor["id"], message),
            )
        except sqlite3.IntegrityError:
            pass  # 重复申请静默处理
    return RedirectResponse("/?applied=1", status_code=302)


# ---------- 管理员页面 ----------
@app.get("/admin", response_class=HTMLResponse)
def admin_home(request: Request, _=Depends(require_admin)):
    with get_db() as conn:
        order_count = conn.execute("SELECT COUNT(*) c FROM orders WHERE status='open'").fetchone()["c"]
        tutor_count = conn.execute("SELECT COUNT(*) c FROM tutors").fetchone()["c"]
        pending_apps = conn.execute("SELECT COUNT(*) c FROM applications WHERE handled=0").fetchone()["c"]
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "order_count": order_count,
        "tutor_count": tutor_count,
        "pending_apps": pending_apps,
    })


@app.get("/admin/import", response_class=HTMLResponse)
def admin_import_page(request: Request, _=Depends(require_admin)):
    return templates.TemplateResponse("admin_import.html", {"request": request})


@app.post("/admin/parse")
def admin_parse(raw_text: str = Form(...), _=Depends(require_admin)):
    """LLM 拆单（不直接入库，先返回让管理员审核）"""
    try:
        orders = parse_orders_from_text(raw_text)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"orders": orders})


@app.post("/admin/import")
def admin_import(orders_json: str = Form(...), _=Depends(require_admin)):
    """审核完毕的单子批量入库"""
    orders = json.loads(orders_json)
    inserted = 0
    with get_db() as conn:
        for o in orders:
            time_slots = o.get("time_slots") or []
            if isinstance(time_slots, str):
                time_slots = [time_slots]
            conn.execute(
                """INSERT INTO orders 
                (level, subject, grade, district, price_per_hour, time_slots, 
                 requirements, parent_contact, raw_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    o.get("level"),
                    o.get("subject"),
                    o.get("grade"),
                    o.get("district"),
                    o.get("price_per_hour"),
                    json.dumps(time_slots, ensure_ascii=False),
                    o.get("requirements", ""),
                    o.get("parent_contact"),
                    o.get("raw_text", ""),
                ),
            )
            inserted += 1
    return RedirectResponse(f"/admin/orders?imported={inserted}", status_code=302)


@app.get("/admin/orders", response_class=HTMLResponse)
def admin_orders(request: Request, _=Depends(require_admin)):
    with get_db() as conn:
        orders = [dict(r) for r in conn.execute(
            "SELECT * FROM orders ORDER BY created_at DESC LIMIT 200"
        ).fetchall()]
    for o in orders:
        try:
            o["time_slots_list"] = json.loads(o["time_slots"]) if o["time_slots"] else []
        except Exception:
            o["time_slots_list"] = []
    return templates.TemplateResponse("admin_orders.html", {"request": request, "orders": orders})


@app.post("/admin/orders/{order_id}/close")
def admin_close_order(order_id: int, _=Depends(require_admin)):
    with get_db() as conn:
        conn.execute("UPDATE orders SET status='closed' WHERE id = ?", (order_id,))
    return RedirectResponse("/admin/orders", status_code=302)


@app.post("/admin/orders/{order_id}/delete")
def admin_delete_order(order_id: int, _=Depends(require_admin)):
    with get_db() as conn:
        conn.execute("DELETE FROM applications WHERE order_id = ?", (order_id,))
        conn.execute("DELETE FROM orders WHERE id = ?", (order_id,))
    return RedirectResponse("/admin/orders", status_code=302)


@app.get("/admin/applications", response_class=HTMLResponse)
def admin_applications(request: Request, _=Depends(require_admin)):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT a.*, o.level, o.subject, o.grade, o.district, o.price_per_hour, 
                   o.parent_contact, o.requirements,
                   t.real_name, t.phone, t.username
            FROM applications a
            JOIN orders o ON a.order_id = o.id
            JOIN tutors t ON a.tutor_id = t.id
            ORDER BY a.handled ASC, a.created_at DESC
        """).fetchall()
        apps = [dict(r) for r in rows]
    return templates.TemplateResponse("admin_applications.html", {"request": request, "apps": apps})


@app.post("/admin/applications/{app_id}/handle")
def admin_handle_app(app_id: int, _=Depends(require_admin)):
    with get_db() as conn:
        conn.execute("UPDATE applications SET handled=1 WHERE id = ?", (app_id,))
    return RedirectResponse("/admin/applications", status_code=302)


@app.get("/admin/tutors", response_class=HTMLResponse)
def admin_tutors(request: Request, _=Depends(require_admin)):
    with get_db() as conn:
        tutors = [dict(r) for r in conn.execute(
            "SELECT * FROM tutors ORDER BY created_at DESC"
        ).fetchall()]
    return templates.TemplateResponse("admin_tutors.html", {"request": request, "tutors": tutors})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
