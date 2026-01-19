"""
数据库模块 - 用户认证、会话管理
"""

from emolagent.database.db import (
    init_db,
    generate_id,
    register_user,
    login_user,
    create_conversation,
    get_user_conversations,
    update_conversation_title,
    delete_conversation,
    add_message,
    get_conversation_messages,
    create_jwt_token,
    verify_jwt_token,
    cleanup_old_data,
    SECRET_KEY,
    DB_FILE,
)

__all__ = [
    "init_db",
    "generate_id",
    "register_user",
    "login_user",
    "create_conversation",
    "get_user_conversations",
    "update_conversation_title",
    "delete_conversation",
    "add_message",
    "get_conversation_messages",
    "create_jwt_token",
    "verify_jwt_token",
    "cleanup_old_data",
    "SECRET_KEY",
    "DB_FILE",
]
