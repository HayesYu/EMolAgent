import os
import json
import time
import datetime
import re
import shutil
from dataclasses import dataclass
from typing import Any

import streamlit as st
import extra_streamlit_components as stx

import database as db
from tools_lib_infer import (
    search_molecule_in_db,
    build_and_optimize_cluster,
    run_dm_infer_pipeline,
    compress_directory,
)

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI


DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnenv.ep154.pth")

WELCOME_MESSAGE = """您好！我是 EMolAgent，您的计算化学 AI 助手。

我专注于分子团簇的自动化建模与电子结构推断。我的工作流涵盖了从本地数据库检索分子、构建并优化团簇结构，到最终预测 HOMO/LUMO、偶极矩及静电势等关键电子性质。

请告诉我您想研究的体系配置，例如：“请构建一个包含 1个Li离子、3个DME分子 和 1个FSI阴离子 的团簇。”

收到指令后，我将为您自动执行查库、建模及计算流程。"""

CUSTOM_SYSTEM_PREFIX = """
你是一个计算化学 AI 助手 EMolAgent。请遵循以下工作流来处理用户的分子计算请求：

1.  **解析需求**：识别用户想要的中心离子（如 Li）、溶剂（如 DME）和阴离子（如 FSI）及其数量。

2.  **数据库检索 (Search_Molecule_DB)**：
    * **优先查库**：对于提到的每个分子（溶剂或阴离子），**必须**先调用 `Search_Molecule_DB` 尝试在本地库中查找。
    * *Solvent* 查 'solvent' 类型，*Salt/Anion* 查 'anion' 类型。
    * **确认反馈**：如果找到了（返回了 `db_path`），告诉用户“已在库中找到 DME (构型已校准)”。如果没找到，则准备使用 SMILES（你需要自己知道或询问用户）。

3.  **建模与优化 (Build_and_Optimize)**：
    * 构造 JSON 参数。
    * 如果第2步找到了 DB 路径，参数里用 `{"name": "DME", "path": "...", "count": 3}`。
    * 如果没找到，用 `{"smiles": "...", "count": 3}`。
    * 此工具会自动进行 UMA 结构优化。

4.  **电子结构推断 (Run_Inference_Pipeline)**：
    * 使用上一步生成的 `optimized_db` 路径。
    * 执行推断并分析性质（HOMO/LUMO/Dipole等）。

5.  **最终报告**：
    * 展示关键的电子性质（如HOMO/LUMO/Dipole/ESP等，从推断结果中读取）。
    * **必须保留** `[[DOWNLOAD:...]]` 链接以便用户下载结果。
    * 最后说明“任务已完成”。

【注意】
* 如果用户说“3个DME”，意思是 count=3。
* FSI 通常是阴离子。
* 一步步执行，不要跳过“查库”步骤，因为库内构型质量最高。
"""

# --- 页面配置 ---
st.set_page_config(page_title="EMolAgent", page_icon="🧪", layout="wide")


# ==============================================================================
# 1. 辅助函数定义
# ==============================================================================

@st.cache_resource(ttl=86400)
def schedule_cleanup():
    """Scheduled cleanup task."""
    db.cleanup_old_data(days=30)
    return True


def get_manager():
    return stx.CookieManager(key="auth_cookie_manager")


def validate_path_exists(path: str, description: str):
    """检查路径是否存在，不存在则终止"""
    if not path or not os.path.exists(path):
        st.error(f"⛔️ **错误：终止执行**\n\n找不到{description}：`{path}`\n\n请检查文件路径是否正确。")
        st.stop()
    return True


def get_user_workspace():
    """根据 session_state 中的用户信息和当前会话ID生成路径"""
    if "user" in st.session_state and st.session_state["user"]:
        username = st.session_state["user"]["username"]
        safe_username = "".join([c for c in username if c.isalnum() or c in ("-", "_")])
        chat_id = st.session_state.get("current_chat_id", "temp")
        workspace = os.path.join("users", safe_username, "output", str(chat_id))
    else:
        workspace = os.path.join("users", "guest", "output", "temp")

    if not os.path.exists(workspace):
        os.makedirs(workspace, exist_ok=True)
    return workspace

def get_user_workspace_from_ids(username: str | None, chat_id: str | None):
    safe_username = "".join([c for c in (username or "guest") if c.isalnum() or c in ("-", "_")])
    safe_chat_id = str(chat_id or "temp")
    workspace = os.path.join("users", safe_username, "output", safe_chat_id)
    os.makedirs(workspace, exist_ok=True)
    return workspace


# ==============================================================================
# 2. Tools
# ==============================================================================

@dataclass
class Context:
    """Custom runtime context schema (可扩展：例如把 user_id 带进 tool runtime)."""
    user_id: str | None = None
    username: str | None = None
    chat_id: str | None = None


@tool(
    "Search_Molecule_DB",
    description=(
        "Search for a molecule (solvent or anion) in the local calibrated database. "
        "Args: query_name (e.g., 'DME'), mol_type ('solvent' or 'anion'). "
        "Returns a string that includes db_path if found."
    ),
)
def tool_search_db(query_name: str, mol_type: str, runtime: ToolRuntime[Context]) -> str:
    """Search molecule in local DB (uses runtime.context for user workspace)."""
    user_ws = get_user_workspace_from_ids(runtime.context.username, runtime.context.chat_id)
    search_dir = os.path.join(user_ws, "search_cache")
    return search_molecule_in_db(query_name, mol_type, search_dir)


@tool(
    "Build_and_Optimize",
    description=(
        "Build a cluster and optimize it using UMA. "
        "Args: ion_name (str), solvents_json (JSON list), anions_json (JSON list). "
        "Each list item should have 'count' and either 'path' or 'smiles'."
    ),
)
def tool_build_optimize(ion_name: str, solvents_json: str, anions_json: str, runtime: ToolRuntime[Context]) -> str:
    """Build+optimize cluster; outputs under the user's workspace."""
    try:
        solvents = json.loads(solvents_json) if solvents_json else []
        anions = json.loads(anions_json) if anions_json else []
    except Exception:
        return "Error parsing JSON inputs."

    user_ws = get_user_workspace_from_ids(runtime.context.username, runtime.context.chat_id)
    task_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
    task_dir = os.path.join(user_ws, f"task_{task_id}")
    return build_and_optimize_cluster(ion_name, solvents, anions, task_dir)


@tool(
    "Run_Inference_Pipeline",
    description=(
        "Run DPTB inference and electronic structure analysis on optimized DB. "
        "Args: optimized_db_path (str), model_path (optional). "
        "Returns a string containing [[DOWNLOAD:...]] zip link on success."
    ),
)
def tool_infer_pipeline(optimized_db_path: str, model_path: str | None = None) -> str:
    """Run inference; returns human-readable result + download marker."""
    if model_path in ["None", "", None]:
        model_path = DEFAULT_MODEL_PATH

    validate_path_exists(optimized_db_path, "Optimized DB")

    db_dir = os.path.dirname(optimized_db_path)
    parent_dir = os.path.dirname(db_dir)

    if os.path.basename(db_dir) == "final_optimized":
        task_root = parent_dir
    elif os.path.basename(db_dir).startswith("task_"):
        task_root = db_dir
    else:
        task_root = db_dir

    run_id = str(time.time_ns())
    infer_out = os.path.join(task_root, f"inference_results_{run_id}")
    result_json_str = run_dm_infer_pipeline(optimized_db_path, model_path, infer_out)

    try:
        res_dict = json.loads(result_json_str)
        if res_dict.get("success"):
            csv_path = res_dict.get("csv_path")
            output_dir = res_dict.get("output_dir", infer_out)
            zip_base_name = os.path.join(task_root, f"analysis_package_{run_id}")
            zip_path = compress_directory(output_dir, zip_base_name)

            return (
                f"推理完成。\n"
                f"CSV摘要路径: {csv_path}\n"
                f"数据预览: {res_dict.get('data_preview')}\n"
                f"[[DOWNLOAD:{zip_path}]]"
            )
        return f"推理出错: {result_json_str}"
    except Exception as e:
        return f"Error processing inference results: {e}"


TOOLS = [tool_search_db, tool_build_optimize, tool_infer_pipeline]


# ==============================================================================
# 3. Agent 初始化
# ==============================================================================

@dataclass
class ResponseFormat:
    """Structured response schema (可选).

    当前 UI 直接展示纯文本 output，并用 [[DOWNLOAD:...]] 做下载。
    因此这里不强制 structured output，只是给未来扩展留接口。
    """
    output: str


@st.cache_resource(show_spinner=False)
def get_checkpointer() -> InMemorySaver:
    # 单机内存 checkpoint：适合 Streamlit demo / 单机部署
    return InMemorySaver()


def build_agent(model_name: str, temperature: float, api_key: str):
    """构建并返回 LangChain agent（每次参数变化时重建）"""
    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        timeout=30,
        max_output_tokens=2000,
    )

    checkpointer = get_checkpointer()

    agent = create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=CUSTOM_SYSTEM_PREFIX,
        context_schema=Context,
        # 如果后面想让 agent 输出结构化结果，可以启用这一行：
        # response_format=ToolStrategy(ResponseFormat),
        checkpointer=checkpointer,
    )
    return agent


# ==============================================================================
# 4. UI：登录 / 主界面
# ==============================================================================

def login_ui(cookie_manager):
    """处理登录和注册的 UI 渲染"""
    st.title("🧪 EMolAgent - 请先登录")
    tab1, tab2 = st.tabs(["登录", "注册"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            submitted = st.form_submit_button("登录")
            if submitted:
                user = db.login_user(username, password)
                if user:
                    st.session_state["user"] = user
                    st.session_state["current_chat_id"] = None
                    st.session_state["logout_flag"] = False
                    token = db.create_jwt_token(user["id"], user["username"])
                    expires = datetime.datetime.now() + datetime.timedelta(days=3)
                    cookie_manager.set("auth_token", token, expires_at=expires)
                    st.success("登录成功！")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("用户名或密码错误")

    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("新用户名")
            new_pass = st.text_input("新密码", type="password")
            confirm_pass = st.text_input("确认密码", type="password")
            submitted = st.form_submit_button("注册")
            if submitted:
                if new_user and new_pass and confirm_pass:
                    if new_pass != confirm_pass:
                        st.error("两次输入的密码不一致")
                    elif db.register_user(new_user, new_pass):
                        st.success("注册成功！请切换到登录标签页进行登录。")
                    else:
                        st.error("用户名已存在")
                else:
                    st.error("请输入用户名和密码")

def normalize_chat_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        try:
            return content.decode("utf-8", errors="replace")
        except Exception:
            return str(content)

    # Gemini/LangChain 有时是 list[dict] 形式的多段内容
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if item is None:
                continue
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, default=str))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p]).strip()

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return json.dumps(content, ensure_ascii=False, default=str)

    return str(content)

def render_message_with_download(role: str, content: Any, key_prefix: str):
    """将 [[DOWNLOAD:...]] 变成可下载按钮，其余文本照常展示"""
    text = normalize_chat_content(content)

    with st.chat_message(role):
        download_match = re.search(r"\[\[DOWNLOAD:(.*?)\]\]", text)
        display_text = re.sub(r"\[\[DOWNLOAD:.*?\]\]", "", text).strip()
        st.write(display_text)

        if download_match:
            file_path = download_match.group(1).strip()
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    st.download_button(
                        label="📦 下载分析结果压缩包 (.zip)",
                        data=f,
                        file_name=os.path.basename(file_path),
                        mime="application/zip",
                        key=f"{key_prefix}_download",
                    )


def main():
    """主函数：包含所有 Streamlit 的 UI 和执行逻辑"""

    # 初始化
    schedule_cleanup()
    cookie_manager = get_manager()

    # --- 认证逻辑 ---
    if "user" not in st.session_state:
        st.session_state["user"] = None

    if st.session_state["user"] is None and not st.session_state.get("logout_flag", False):
        token = cookie_manager.get("auth_token")
        if token:
            user_info = db.verify_jwt_token(token)
            if user_info:
                st.session_state["user"] = user_info
                st.session_state["current_chat_id"] = None

    # 如果未登录，显示登录页并停止
    if st.session_state["user"] is None:
        login_ui(cookie_manager)
        return

    current_user = st.session_state["user"]
    if "suppress_autocreate" not in st.session_state:
        st.session_state["suppress_autocreate"] = False

    # 1. Sidebar
    with st.sidebar:
        st.write(f"👤 **{current_user['username']}**")
        if st.button("登出", type="secondary"):
            st.session_state["user"] = None
            st.session_state["messages"] = []
            st.session_state["current_chat_id"] = None
            st.session_state["logout_flag"] = True
            cookie_manager.delete("auth_token")
            st.rerun()

        st.markdown("---")
        if st.button("➕ 新建对话", type="primary", use_container_width=True):
            st.session_state["suppress_autocreate"] = False
            new_id = db.create_conversation(current_user["id"], title="New Chat")
            st.session_state["current_chat_id"] = new_id
            st.session_state["messages"] = [{"role": "assistant", "content": WELCOME_MESSAGE}]
            db.add_message(new_id, "assistant", WELCOME_MESSAGE)
            st.rerun()

        st.markdown("### 🕒 历史记录")
        conversations = db.get_user_conversations(current_user["id"])
        for chat in conversations:
            btn_type = "primary" if st.session_state.get("current_chat_id") == chat["id"] else "secondary"
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if st.button(
                    f"📄 {chat['title']}",
                    key=f"chat_{chat['id']}",
                    type=btn_type,
                    use_container_width=True,
                ):
                    st.session_state["current_chat_id"] = chat["id"]
                    msgs = db.get_conversation_messages(chat["id"])
                    st.session_state["messages"] = msgs if msgs else []
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{chat['id']}"):
                    if len(conversations) == 1:
                        st.session_state["suppress_autocreate"] = True
                        st.session_state["current_chat_id"] = None
                        st.session_state["messages"] = []
                    safe_username = "".join([c for c in current_user["username"] if c.isalnum() or c in ("-", "_")])
                    chat_folder = os.path.join("users", safe_username, "output", str(chat["id"]))
                    if os.path.exists(chat_folder):
                        try:
                            shutil.rmtree(chat_folder)
                        except Exception as e:
                            st.toast(f"⚠️ 文件夹删除失败: {e}")
                    db.delete_conversation(chat["id"])
                    if st.session_state.get("current_chat_id") == chat["id"]:
                        st.session_state["current_chat_id"] = None
                        st.session_state["messages"] = []
                    st.rerun()

        st.markdown("---")
        st.header("模型设置")
        model_name = st.selectbox(
            "选择模型",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-exp"],
            index=0,
        )
        api_key = os.getenv("GOOGLE_API_KEY", "")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0)

    # 2. Session Init
    if st.session_state.get("current_chat_id") is None:
        user_conversations = db.get_user_conversations(current_user["id"])
        if user_conversations:
            latest_chat = user_conversations[0]
            st.session_state["current_chat_id"] = latest_chat["id"]
            msgs = db.get_conversation_messages(latest_chat["id"])
            st.session_state["messages"] = msgs if msgs else []
        else:
            if st.session_state.get("suppress_autocreate"):
                st.session_state["current_chat_id"] = None
                st.session_state["messages"] = []
            else:
                new_id = db.create_conversation(current_user["id"], title="New Chat")
                st.session_state["current_chat_id"] = new_id
                st.session_state["messages"] = [{"role": "assistant", "content": WELCOME_MESSAGE}]
                db.add_message(new_id, "assistant", WELCOME_MESSAGE)

    if st.session_state.get("current_chat_id") is None:
        st.title("🧪 EMolAgent")
        st.info("暂无对话，请在左侧点击“➕ 新建对话”。")
        return

    # 3. LLM Setup
    if not api_key:
        st.warning("⚠️ Google API Key 无效。")
        st.stop()

    try:
        agent = build_agent(model_name=model_name, temperature=temperature, api_key=api_key)
    except Exception as e:
        st.error(f"模型/Agent 初始化失败: {e}")
        st.stop()

    # 4. Chat Interface
    st.title("🧪 EMolAgent")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for idx, msg in enumerate(st.session_state["messages"]):
        render_message_with_download(
            role=msg["role"],
            content=msg["content"],
            key_prefix=f"history_{idx}",
        )

    # 5. Handle Input
    if prompt_input := st.chat_input("请输入指令..."):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        st.chat_message("user").write(prompt_input)

        current_chat_id = st.session_state["current_chat_id"]
        db.add_message(current_chat_id, "user", prompt_input)

        if len(st.session_state.messages) <= 2:
            db.update_conversation_title(current_chat_id, prompt_input[:20])

        # LangChain new agent expects {"messages": [...]} style
        # 并且可以配合 checkpointer 使用 thread_id 来维持同一对话的短期记忆
        config = {"configurable": {"thread_id": str(current_chat_id)}}
        context = Context(
            user_id=str(current_user.get("id")) if current_user else None,
            username=current_user.get("username") if current_user else None,
            chat_id=str(current_chat_id),
        )

        with st.spinner("正在思考和执行任务..."):
            try:
                response: dict[str, Any] = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt_input}]},
                    config=config,
                    context=context,
                )

                # create_agent 的返回通常是一个 dict，里面含 messages。
                # 这里做一个稳健提取：优先取最后一条 assistant message 的 content。
                output_text = None
                msgs = response.get("messages") if isinstance(response, dict) else None
                if msgs and isinstance(msgs, list):
                    # msgs 里可能是 dict 或 BaseMessage；都做兼容
                    last = msgs[-1]
                    if isinstance(last, dict):
                        output_text = last.get("content")
                    else:
                        output_text = getattr(last, "content", None)

                # 兜底：如果模型返回 structured_response 或 output 字段
                if not output_text and isinstance(response, dict):
                    output_text = response.get("output") or response.get("structured_response")

                if not output_text:
                    output_text = str(response)

                output_text_str = normalize_chat_content(output_text)

                render_message_with_download(
                    role="assistant",
                    content=output_text_str,
                    key_prefix="current_run",
                )

                st.session_state.messages.append({"role": "assistant", "content": output_text_str})
                db.add_message(current_chat_id, "assistant", output_text_str)

            except Exception as e:
                error_msg = f"执行出错: {str(e)}"
                render_message_with_download(
                    role="assistant",
                    content=error_msg,
                    key_prefix="current_run_error",
                )
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                db.add_message(current_chat_id, "assistant", error_msg)


# ==============================================================================
# 5. 程序入口保护
# ==============================================================================

if __name__ == "__main__":
    main()