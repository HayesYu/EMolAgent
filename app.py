import os
import json
import time
import datetime
import re
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import importlib

def _import_attr(attr_name: str, module_candidates: list[str]):
    last_err = None
    for module_name in module_candidates:
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, attr_name):
                return getattr(mod, attr_name)
        except Exception as e:
            last_err = e
    raise ImportError(
        f"Cannot import {attr_name} from any of {module_candidates}. Last error: {last_err}"
    )

# 1) AgentExecutor：不同版本所在位置不一样
AgentExecutor = _import_attr(
    "AgentExecutor",
    [
        "langchain.agents",
        "langchain.agents.agent",
        "langchain.agents.agent_executor",
        "langchain.agents.executor",
    ],
)

# 2) create_tool_calling_agent：找不到就退化到 create_react_agent（保证先能跑起来）
try:
    create_tool_calling_agent = _import_attr(
        "create_tool_calling_agent",
        [
            "langchain.agents",
            "langchain.agents.tool_calling_agent.base",
            "langchain.agents.tool_calling_agent",
        ],
    )
except ImportError:
    create_react_agent = _import_attr(
        "create_react_agent",
        [
            "langchain.agents",
            "langchain.agents.react.agent",
            "langchain.agents.react.base",
        ],
    )

    def create_tool_calling_agent(llm, tools, prompt):
        return create_react_agent(llm, tools, prompt)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import StructuredTool
import extra_streamlit_components as stx
import shutil
from tools_lib_infer import (
    search_molecule_in_db, 
    build_and_optimize_cluster, 
    run_dm_infer_pipeline, 
    compress_directory
)
import database as db

# --- 全局常量定义 (保持在顶层) ---
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnenv.ep154.pth")

WELCOME_MESSAGE = """您好！我是 EMolAgent，您的计算化学 AI 助手。

我专注于分子团簇的自动化建模与电子结构推断。我的工作流涵盖了从本地数据库检索分子、构建并优化团簇结构，到最终预测 HOMO/LUMO、偶极矩及静电势等关键性质。

请告诉我您想研究的体系配置，例如：“请构建一个包含 1个Li离子、3个DME分子 和 1个FSI阴离子 的团簇。”

收到指令后，我将为您自动执行查库、建模及计算流程。"""

CUSTOM_SYSTEM_PREFIX = """
你是一个计算化学 AI 助手 EMolAgent。请遵循以下工作流来处理用户的分子计算请求：

1.  **解析需求**：识别用户想要的中心离子（如 Li）、溶剂（如 DME）和阴离子（如 FSI）及其数量。

2.  **数据库检索 (Search_Molecule_DB)**：
    * **优先查库**：对于提到的每个分子（溶剂或阴离子），**必须**先调用 `Search_Molecule_DB` 尝试在本地库中查找。
    * *Solvent* 查 'solvent' 类型，*Salt/Anion* 查 'anion' 类型。
    * **确认反馈**：如果找到了（返回了 `db_path`），告诉用户“已在库中找到 DME (构型已校准)”。如果没找到，则准备使用 SMILES（你需要自己知道或询问用户 SMILES，常用分子如 DME=COCCOC 可自备）。

3.  **建模与优化 (Build_and_Optimize)**：
    * 构造 JSON 参数。
    * 如果第2步找到了 DB 路径，参数里用 `{{"name": "DME", "path": "...", "count": 3}}`。
    * 如果没找到，用 `{{"smiles": "...", "count": 3}}`。
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

# --- 页面配置 (必须是第一个 Streamlit 命令) ---
st.set_page_config(page_title="EMolAgent", page_icon="🧪", layout="wide")


# ==============================================================================
# 1. 辅助函数定义 (只定义，不执行)
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
        safe_username = "".join([c for c in username if c.isalnum() or c in ('-','_')])
        chat_id = st.session_state.get("current_chat_id", "temp")
        workspace = os.path.join("users", safe_username, "output", str(chat_id))
    else:
        workspace = os.path.join("users", "guest", "output", "temp")
    
    if not os.path.exists(workspace):
        os.makedirs(workspace, exist_ok=True)
    return workspace

# --- Tool Functions (被 Agent 调用) ---

def tool_search_db(query_name: str, mol_type: str):
    user_ws = get_user_workspace()
    search_dir = os.path.join(user_ws, "search_cache")
    return search_molecule_in_db(query_name, mol_type, search_dir)

def tool_build_optimize(ion_name: str, solvents_json: str, anions_json: str):
    try:
        solvents = json.loads(solvents_json) if solvents_json else []
        anions = json.loads(anions_json) if anions_json else []
    except:
        return "Error parsing JSON inputs."

    user_ws = get_user_workspace()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    task_dir = os.path.join(user_ws, f"task_{timestamp}")
    return build_and_optimize_cluster(ion_name, solvents, anions, task_dir)

def tool_infer_pipeline(optimized_db_path: str, model_path: str = None):
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

    infer_out = os.path.join(task_root, "inference_results")
    result_json_str = run_dm_infer_pipeline(optimized_db_path, model_path, infer_out)
    
    try:
        res_dict = json.loads(result_json_str)
        if res_dict.get("success"):
            csv_path = res_dict.get("csv_path")
            output_dir = res_dict.get("output_dir", infer_out)
            zip_base_name = os.path.join(task_root, "analysis_package")
            zip_path = compress_directory(output_dir, zip_base_name)
            
            return (
                f"推理完成。\n"
                f"CSV摘要路径: {csv_path}\n"
                f"数据预览: {res_dict.get('data_preview')}\n"
                f"[[DOWNLOAD:{zip_path}]]"
            )
        else:
            return f"推理出错: {result_json_str}"
    except Exception as e:
        return f"Error processing inference results: {e}"

# --- Tool 定义列表 (静态定义) ---
tools = [
    StructuredTool.from_function(
        func=tool_search_db,
        name="Search_Molecule_DB",
        description="Search for a molecule (solvent or anion) in the local calibrated database. Returns a DB path if found. Args: query_name (e.g., 'DME'), mol_type ('solvent' or 'anion')."
    ),
    StructuredTool.from_function(
        func=tool_build_optimize,
        name="Build_and_Optimize",
        description="Build a cluster and optimize it using UMA. Provide solvents/anions config as JSON lists. Each item should have 'count', and either 'path' (from Search tool) or 'smiles'. Example: solvents_json='[{\"name\":\"DME\", \"path\":\"...db\", \"count\":3}]'"
    ),
    StructuredTool.from_function(
        func=tool_infer_pipeline,
        name="Run_Inference_Pipeline",
        description="Run DPTB inference and Electronic Structure Analysis on the optimized DB. Args: optimized_db_path."
    )
]


# ==============================================================================
# 2. 主要 UI 逻辑封装
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
        return # 替代 st.stop() 以便结构更清晰，不过在 main 中 return 等同于结束

    # --- 已登录主界面 ---
    current_user = st.session_state["user"]

    # 1. Sidebar
    with st.sidebar:
        st.write(f"👤 **{current_user['username']}**") # 这一行之前报错，现在因为在 main 中且已登录，所以安全
        if st.button("登出", type="secondary"):
            st.session_state["user"] = None
            st.session_state["messages"] = []
            st.session_state["current_chat_id"] = None
            st.session_state["logout_flag"] = True
            cookie_manager.delete("auth_token")
            st.rerun()
        
        st.markdown("---")
        if st.button("➕ 新建对话", type="primary", use_container_width=True):
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
                if st.button(f"📄 {chat['title']}", key=f"chat_{chat['id']}", type=btn_type, use_container_width=True):
                    st.session_state["current_chat_id"] = chat["id"]
                    msgs = db.get_conversation_messages(chat["id"])
                    st.session_state["messages"] = msgs if msgs else []
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{chat['id']}"):
                    safe_username = "".join([c for c in current_user['username'] if c.isalnum() or c in ('-','_')])
                    chat_folder = os.path.join("users", safe_username, "output", str(chat['id']))
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
        model_name = st.selectbox("选择模型", ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"], index=0)
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
            new_id = db.create_conversation(current_user["id"], title="New Chat")
            st.session_state["current_chat_id"] = new_id
            st.session_state["messages"] = [{"role": "assistant", "content": WELCOME_MESSAGE}]
            db.add_message(new_id, "assistant", WELCOME_MESSAGE)

    # 3. LLM Setup
    if not api_key:
        st.warning("⚠️ Google API Key 无效。")
        st.stop()

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )
    except Exception as e:
        st.error(f"模型连接失败: {e}")
        st.stop()

    # Agent Prompt & Executor (在此处初始化，避免子进程执行)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PREFIX),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)

    # 4. Chat Interface
    st.title("🧪 EMolAgent")

    for idx, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            content = msg["content"]
            download_match = re.search(r"\[\[DOWNLOAD:(.*?)\]\]", content)
            display_text = re.sub(r"\[\[DOWNLOAD:.*?\]\]", "", content).strip()
            st.write(display_text)
            if download_match:
                file_path = download_match.group(1)
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label="📦 下载分析结果压缩包 (.zip)",
                            data=f,
                            file_name=os.path.basename(file_path),
                            mime="application/zip",
                            key=f"history_btn_{idx}"
                        )

    # 5. Handle Input
    if prompt_input := st.chat_input("请输入指令..."):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        st.chat_message("user").write(prompt_input)
        
        current_chat_id = st.session_state["current_chat_id"]
        db.add_message(current_chat_id, "user", prompt_input)
        
        if len(st.session_state.messages) <= 2: 
            db.update_conversation_title(current_chat_id, prompt_input[:20])

        history_langchain = []
        for m in st.session_state["messages"][:-1]:
            if m["role"] == "user":
                history_langchain.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                history_langchain.append(AIMessage(content=m["content"]))

        with st.chat_message("assistant"):
            with st.spinner("正在思考和执行任务..."):
                try:
                    response = agent_executor.invoke(
                        {
                            "input": prompt_input,
                            "chat_history": history_langchain 
                        }
                    )
                    output_text = response["output"]
                    download_match = re.search(r"\[\[DOWNLOAD:(.*?)\]\]", output_text)
                    clean_text = re.sub(r"\[\[DOWNLOAD:.*?\]\]", "", output_text).strip()
                    st.write(clean_text)
                    
                    if download_match:
                        file_path = download_match.group(1)
                        if os.path.exists(file_path):
                            with open(file_path, "rb") as f:
                                st.download_button(
                                    label="📦 下载分析结果压缩包 (.zip)",
                                    data=f,
                                    file_name=os.path.basename(file_path),
                                    mime="application/zip",
                                    key="current_run_btn"
                                )
                    
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                    db.add_message(current_chat_id, "assistant", output_text)
                    
                except Exception as e:
                    error_msg = f"执行出错: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    db.add_message(current_chat_id, "assistant", error_msg)

# ==============================================================================
# 3. 程序入口保护 (Crucial for Multiprocessing)
# ==============================================================================

if __name__ == "__main__":
    main()