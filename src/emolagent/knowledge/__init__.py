"""
知识库模块

提供基于 ChromaDB + Google Embedding 的 RAG 文献检索功能。
"""

from emolagent.knowledge.knowledge_base import (
    search_knowledge,
    build_index,
    get_index_stats,
    LITERATURE_PATH,
    CHROMA_DB_PATH,
    get_vectorstore,
    get_chroma_client,
)

__all__ = [
    "search_knowledge",
    "build_index",
    "get_index_stats",
    "LITERATURE_PATH",
    "CHROMA_DB_PATH",
    "get_vectorstore",
    "get_chroma_client",
]
