"""
知识库管理模块 - 基于 ChromaDB + Google Embedding 的 RAG 实现
支持 PDF、Markdown、PPTX 文献的索引与检索
"""

import re
import time
import os
import hashlib
import json
import sys
import shutil
from typing import List, Optional
from pathlib import Path

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
)
from langchain_chroma import Chroma


# ==========================================
# 常量定义
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
CHROMA_DB_PATH = os.getenv("EMOL_CHROMA_DB_PATH", DEFAULT_CHROMA_DB_PATH)
LITERATURE_PATH = "/home/hayes/projects/ai4mol"
INDEX_STATE_FILE = os.path.join(CHROMA_DB_PATH, "index_state.json")

COLLECTION_NAME = "ai4mol_literature"

def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    # 去掉常见的 NUL 字符
    text = text.replace("\x00", "")
    # 丢弃 surrogate 等非法 UTF-8 字符
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return text.strip()

def _ensure_writable_dir(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)
    test_path = os.path.join(dir_path, ".write_test")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("ok")
    os.remove(test_path)

def get_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    """获取 Google Embedding 模型"""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
        task_type="retrieval_document",
    )


def get_query_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    """获取用于查询的 Embedding 模型"""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
        task_type="retrieval_query",
    )


def reset_chroma_client():
    return

def get_chroma_client() -> chromadb.PersistentClient:
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    _ensure_writable_dir(CHROMA_DB_PATH)
    return chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )


def get_vectorstore(api_key: str) -> Chroma:
    """获取或创建 Chroma 向量数据库"""
    embeddings = get_embeddings(api_key)
    client = get_chroma_client()
    
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        client=client,
    )
    return vectorstore


def compute_file_hash(file_path: str) -> str:
    """计算文件的 MD5 哈希值，用于增量索引"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_index_state() -> dict:
    """加载已索引文件的状态"""
    if os.path.exists(INDEX_STATE_FILE):
        try:
            with open(INDEX_STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_index_state(state: dict):
    """保存索引状态"""
    os.makedirs(os.path.dirname(INDEX_STATE_FILE), exist_ok=True)
    with open(INDEX_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def collect_documents(literature_path: str) -> List[str]:
    """
    递归收集所有支持的文档路径
    支持: .pdf, .md, .pptx
    """
    supported_extensions = {'.pdf', '.md', '.pptx'}
    documents = []
    
    if not os.path.exists(literature_path):
        print(f"Warning: Literature path does not exist: {literature_path}")
        return documents
    
    for root, dirs, files in os.walk(literature_path):
        # 跳过隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                documents.append(os.path.join(root, file))
    
    return documents


def load_document(file_path: str) -> List:
    """根据文件类型加载文档"""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.md':
            loader = UnstructuredMarkdownLoader(file_path)
        elif ext == '.pptx':
            loader = UnstructuredPowerPointLoader(file_path)
        else:
            return []
        
        docs = loader.load()
        cleaned_docs = []
        for doc in docs:
            doc.page_content = _clean_text(getattr(doc, "page_content", ""))
            if not doc.page_content:
                continue
            cleaned_docs.append(doc)
        docs = cleaned_docs
        
        # 添加元数据
        for doc in docs:
            doc.metadata['source'] = file_path
            doc.metadata['filename'] = os.path.basename(file_path)
            # 提取相对路径作为分类
            rel_path = os.path.relpath(file_path, LITERATURE_PATH)
            doc.metadata['category'] = os.path.dirname(rel_path)
        
        return docs
    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return []


def build_index(
    api_key: str,
    literature_path: str = LITERATURE_PATH,
    force_rebuild: bool = False,
    progress_callback=None
) -> dict:
    """
    构建或增量更新知识库索引
    """
    
    stats = {
        "total_files": 0,
        "new_indexed": 0,
        "skipped": 0,
        "failed": 0,
        "total_chunks": 0
    }
    
    # 收集所有文档
    all_files = collect_documents(literature_path)
    stats["total_files"] = len(all_files)
    
    if not all_files:
        return stats
    
    # 如果强制重建，先彻底清理
    if force_rebuild:
        client = get_chroma_client()
        client.reset()
        
        if os.path.exists(INDEX_STATE_FILE):
            os.remove(INDEX_STATE_FILE)

        index_state = {}
    else:
        index_state = load_index_state()
    
    # 获取向量数据库
    vectorstore = get_vectorstore(api_key)
    
    # 文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    # ----------------------------
    # 进度统计（每 N 个文件打印一次）
    # ----------------------------
    log_every_n = int(os.getenv("EMOL_INDEX_LOG_EVERY_N", "20"))
    start_ts = time.time()
    last_log_ts = start_ts
    last_log_processed = 0
    last_log_chunks = 0

    def _fmt_eta(seconds: float) -> str:
        if seconds < 0 or seconds == float("inf"):
            return "未知"
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h}h{m:02d}m{s:02d}s"
        if m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"
    # 处理每个文件
    for idx, file_path in enumerate(all_files):
        if progress_callback:
            progress_callback(idx + 1, len(all_files), os.path.basename(file_path))
        
        try:
            file_hash = compute_file_hash(file_path)
            
            # 检查是否需要更新
            if file_path in index_state and index_state[file_path] == file_hash:
                stats["skipped"] += 1
                continue
            
            # 加载文档
            docs = load_document(file_path)
            if not docs:
                stats["failed"] += 1
                continue
            
            # 分割文档
            chunks = text_splitter.split_documents(docs)
            cleaned_chunks = []
            for c in chunks:
                c.page_content = _clean_text(getattr(c, "page_content", ""))
                if not c.page_content:
                    continue
                cleaned_chunks.append(c)
            chunks = cleaned_chunks

            if not chunks:
                stats["failed"] += 1
                continue
            
            if chunks:
                # 如果是更新，先删除旧的
                if file_path in index_state:
                    try:
                        collection = vectorstore._collection
                        collection.delete(where={"source": file_path})
                    except Exception:
                        pass
                
                # 添加新向量
                vectorstore.add_documents(chunks)
                stats["total_chunks"] += len(chunks)
                stats["new_indexed"] += 1
                
                # 更新索引状态
                index_state[file_path] = file_hash
                
                # 每处理10个文件保存一次状态
                if stats["new_indexed"] % 10 == 0:
                    save_index_state(index_state)
        
        except Exception as e:
            safe_err = str(e).encode("utf-8", "backslashreplace").decode("utf-8")
            print(f"Error processing {file_path}: {safe_err}")
            stats["failed"] += 1
        finally:
            processed = idx + 1
            if log_every_n > 0 and (processed % log_every_n == 0 or processed == len(all_files)):
                now = time.time()
                elapsed = max(now - start_ts, 1e-6)
                interval = max(now - last_log_ts, 1e-6)

                files_per_s = processed / elapsed
                chunks_total = stats["total_chunks"]
                chunks_per_s = chunks_total / elapsed

                # 用“平均每文件耗时”估 ETA
                remaining_files = len(all_files) - processed
                avg_sec_per_file = elapsed / processed
                eta_sec = remaining_files * avg_sec_per_file

                # 区间速率（看最近 N 个文件的速度）
                interval_files = processed - last_log_processed
                interval_chunks = chunks_total - last_log_chunks
                interval_files_per_s = interval_files / interval
                interval_chunks_per_s = interval_chunks / interval

                print(
                    "[KB] 进度: "
                    f"{processed}/{len(all_files)} files, "
                    f"indexed={stats['new_indexed']}, skipped={stats['skipped']}, failed={stats['failed']}, "
                    f"chunks={chunks_total} | "
                    f"avg={files_per_s:.2f} files/s, {chunks_per_s:.1f} chunks/s | "
                    f"recent={interval_files_per_s:.2f} files/s, {interval_chunks_per_s:.1f} chunks/s | "
                    f"ETA={_fmt_eta(eta_sec)}"
                )

                last_log_ts = now
                last_log_processed = processed
                last_log_chunks = chunks_total
    
    # 保存索引状态
    save_index_state(index_state)
    
    return stats


def search_knowledge(
    query: str,
    api_key: str,
    top_k: int = 5,
    filter_category: Optional[str] = None
) -> List[dict]:
    """在知识库中搜索相关内容"""
    embeddings = get_query_embeddings(api_key)
    client = get_chroma_client()
    
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        client=client,
    )
    
    # 构建过滤条件
    search_kwargs = {"k": top_k}
    if filter_category:
        search_kwargs["filter"] = {"category": filter_category}
    
    # 执行相似度搜索
    results = vectorstore.similarity_search_with_relevance_scores(
        query, **search_kwargs
    )
    
    # 格式化结果
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("filename", "Unknown"),
            "category": doc.metadata.get("category", ""),
            "full_path": doc.metadata.get("source", ""),
            "relevance_score": round(score, 4)
        })
    
    return formatted_results


def get_index_stats(api_key: str) -> dict:
    """获取知识库统计信息"""
    try:
        # 先检查目录是否存在
        if not os.path.exists(CHROMA_DB_PATH):
            return {
                "total_documents": 0,
                "indexed_files": 0
            }
        
        client = get_chroma_client()
        
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            total_docs = collection.count()
        except Exception:
            total_docs = 0
        
        return {
            "total_documents": total_docs,
            "index_state_file": INDEX_STATE_FILE,
            "indexed_files": len(load_index_state())
        }
    except Exception as e:
        return {"error": str(e)}