"""
数据库操作模块

提供用户认证、会话管理、消息存储等功能。
"""

import sqlite3
import bcrypt
import os
import jwt
import datetime
import time
import threading

from emolagent.utils.logger import logger
from emolagent.utils.paths import get_data_path


class SimpleSnowflake:
    """雪花算法 ID 生成器。"""
    
    def __init__(self, datacenter_id=1, worker_id=1):
        self.datacenter_id = datacenter_id
        self.worker_id = worker_id
        self.sequence = 0
        self.last_timestamp = -1
        
        self.timestamp_shift = 22
        self.datacenter_id_shift = 17
        self.worker_id_shift = 12
        self.sequence_mask = 0xFFF
        self.lock = threading.Lock()

    def _current_timestamp(self):
        return int(time.time() * 1000)

    def next_id(self):
        with self.lock:
            timestamp = self._current_timestamp()

            if timestamp < self.last_timestamp:
                raise Exception("Clock moved backwards!")

            if self.last_timestamp == timestamp:
                self.sequence = (self.sequence + 1) & self.sequence_mask
                if self.sequence == 0:
                    while timestamp <= self.last_timestamp:
                        timestamp = self._current_timestamp()
            else:
                self.sequence = 0

            self.last_timestamp = timestamp

            new_id = ((timestamp << self.timestamp_shift) |
                      (self.datacenter_id << self.datacenter_id_shift) |
                      (self.worker_id << self.worker_id_shift) |
                      self.sequence)
            return new_id


# 初始化生成器
id_generator = SimpleSnowflake()


def generate_id():
    """生成唯一 ID。"""
    return id_generator.next_id()


SECRET_KEY = "TwinkleTwinkleLittleStar"
DB_FILE = get_data_path("chat_app.db")


def init_db():
    """初始化数据库表结构。"""
    # 确保 data 目录存在
    data_dir = os.path.dirname(DB_FILE)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    conn = sqlite3.connect(DB_FILE, timeout=30)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    
    # 用户表
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                )''')
    
    # 会话表
    c.execute('''CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )''')
    
    try:
        c.execute("ALTER TABLE conversations ADD COLUMN updated_at TIMESTAMP")
    except sqlite3.OperationalError:
        pass

    # 消息表
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY,
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
    """注册新用户。"""
    conn = sqlite3.connect(DB_FILE, timeout=30)
    c = conn.cursor()
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    uid = generate_id()
    try:
        c.execute("INSERT INTO users (id, username, password_hash) VALUES (?, ?, ?)", (uid, username, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def login_user(username, password):
    """用户登录，返回 user_id 和 username。"""
    conn = sqlite3.connect(DB_FILE, timeout=30)
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
    """创建新会话。"""
    conn = sqlite3.connect(DB_FILE, timeout=30)
    c = conn.cursor()
    cid = generate_id()
    c.execute("INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)", 
              (cid, user_id, title))
    conn.commit()
    conn.close()
    return cid


def get_user_conversations(user_id):
    """获取用户的所有会话列表。"""
    conn = sqlite3.connect(DB_FILE, timeout=30)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "date": r[2]} for r in rows]


def update_conversation_title(conversation_id, new_title):
    """更新会话标题。"""
    conn = sqlite3.connect(DB_FILE, timeout=30)
    c = conn.cursor()
    c.execute("UPDATE conversations SET title = ? WHERE id = ?", (new_title, conversation_id))
    conn.commit()
    conn.close()


def delete_conversation(conversation_id):
    """删除会话及其消息。"""
    conn = sqlite3.connect(DB_FILE, timeout=30)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    c.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()


# --- 消息管理 ---

def add_message(conversation_id, role, content):
    """添加一条消息。"""
    conn = sqlite3.connect(DB_FILE, timeout=30)
    c = conn.cursor()
    mid = generate_id()
    c.execute("INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?)", 
              (mid, conversation_id, role, content))
    c.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()


def get_conversation_messages(conversation_id):
    """获取某会话的所有消息。"""
    conn = sqlite3.connect(DB_FILE, timeout=30)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC", (conversation_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]


def create_jwt_token(user_id, username):
    """生成 JWT Token。"""
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=3)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token


def verify_jwt_token(token):
    """验证 Token。"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return {"id": payload["user_id"], "username": payload["username"]}
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def cleanup_old_data(days=30):
    """清理旧数据。"""
    conn = sqlite3.connect(DB_FILE, timeout=30)
    c = conn.cursor()
    
    cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    
    try:
        c.execute("DELETE FROM messages WHERE timestamp < ?", (cutoff_date,))
        c.execute("""
            DELETE FROM conversations 
            WHERE COALESCE(updated_at, created_at) < ?
        """, (cutoff_date,))
        c.execute("DELETE FROM messages WHERE conversation_id NOT IN (SELECT id FROM conversations)")
        conn.commit()
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
    finally:
        conn.close()


# 模块加载时初始化数据库
init_db()
