import sqlite3
import bcrypt
import os
from datetime import datetime
import jwt
import datetime

SECRET_KEY = "TwinkleTwinkleLittleStar"
DB_FILE = "data/chat_app.db"

def init_db():
    """初始化数据库表结构"""
    if not os.path.exists("data"):
        os.makedirs("data")
        
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 用户表
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                )''')
    
    # 会话表 (Conversations)
    c.execute('''CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )''')
    
    # 消息表 (Messages)
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                )''')
    
    conn.commit()
    conn.close()

# --- 用户认证 ---

def register_user(username, password):
    """注册新用户"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # 用户名已存在
    finally:
        conn.close()

def login_user(username, password):
    """用户登录，返回 user_id 和 username"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    data = c.fetchone()
    conn.close()
    
    if data:
        user_id, stored_hash = data
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return {"id": user_id, "username": username}
    return None

# --- 会话管理 ---

def create_conversation(user_id, title="New Chat"):
    """创建新会话"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO conversations (user_id, title) VALUES (?, ?)", (user_id, title))
    cid = c.lastrowid
    conn.commit()
    conn.close()
    return cid

def get_user_conversations(user_id):
    """获取用户的所有会话列表"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "date": r[2]} for r in rows]

def update_conversation_title(conversation_id, new_title):
    """更新会话标题（通常用第一句话作为标题）"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE conversations SET title = ? WHERE id = ?", (new_title, conversation_id))
    conn.commit()
    conn.close()

def delete_conversation(conversation_id):
    """删除会话及其消息"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    c.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()

# --- 消息管理 ---

def add_message(conversation_id, role, content):
    """添加一条消息"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)", 
              (conversation_id, role, content))
    conn.commit()
    conn.close()

def get_conversation_messages(conversation_id):
    """获取某会话的所有消息"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC", (conversation_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]

def create_jwt_token(user_id, username):
    """生成 JWT Token"""
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=3) # 过期时间
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

def verify_jwt_token(token):
    """验证 Token，成功返回用户信息，失败返回 None"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return {"id": payload["user_id"], "username": payload["username"]}
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# 初始化数据库
init_db()