import os
import json
import time
import datetime
import re
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import StructuredTool
from langchain.callbacks import StreamlitCallbackHandler
import extra_streamlit_components as stx
import shutil
from tools_lib_infer import (
    search_molecule_in_db, 
    build_and_optimize_cluster, 
    run_dm_infer_pipeline, 
    compress_directory
)
import database as db

#os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
#os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
#os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnenv.ep154.pth")

# --- 页面配置 ---
st.set_page_config(page_title="EMolAgent", page_icon="🧪", layout="wide")

db.cleanup_old_data(days=30)

def get_manager():
    return stx.CookieManager(key="auth_cookie_manager")

cookie_manager = get_manager()

# ==============================================================================
# 1. 认证模块 (登录/注册 UI)
# ==============================================================================

if "user" not in st.session_state:
    st.session_state["user"] = None

# 尝试从 Cookie 恢复会话
if st.session_state["user"] is None and not st.session_state.get("logout_flag", False):
    # 获取 cookie 中的 token
    token = cookie_manager.get("auth_token")
    if token:
        user_info = db.verify_jwt_token(token)
        if user_info:
            st.session_state["user"] = user_info
            st.session_state["current_chat_id"] = None # 或者恢复上次的会话ID

def login_page():
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
                    st.session_state["current_chat_id"] = None # 登录后重置当前会话
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

# 如果未登录，只显示登录页并停止后续执行
if st.session_state["user"] is None:
    login_page()
    st.stop()

# ==============================================================================
# 2. 已登录的主界面逻辑
# ==============================================================================

# 获取当前用户信息
current_user = st.session_state["user"]

# --- 侧边栏与控制 ---
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

    # 新建会话按钮
    if st.button("➕ 新建对话", type="primary", use_container_width=True):
        new_id = db.create_conversation(current_user["id"], title="New Chat")
        st.session_state["current_chat_id"] = new_id
        st.session_state["messages"] = [{"role": "assistant", "content": "你好！我是你的 AI 助手，全自动分子分析模式已就绪。"}]
        # 存入初始欢迎语到数据库
        db.add_message(new_id, "assistant", "你好！我是你的 AI 助手，全自动分子分析模式已就绪。")
        st.rerun()

    st.markdown("### 🕒 历史记录")

    # 获取并显示会话列表
    conversations = db.get_user_conversations(current_user["id"])
    for chat in conversations:
        # 简单的样式处理，高亮当前选中的会话
        btn_type = "primary" if st.session_state.get("current_chat_id") == chat["id"] else "secondary"
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(f"📄 {chat['title']}", key=f"chat_{chat['id']}", type=btn_type, use_container_width=True):
                st.session_state["current_chat_id"] = chat["id"]
                # 从数据库加载历史消息
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
    model_name = st.selectbox(
        "选择模型", 
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-exp"], 
        index=0
    )
    # api_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", "")) # 尝试读取环境变量
    api_key = os.getenv("GOOGLE_API_KEY", "")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0)

# --- 检查会话状态 ---
# 如果进入主界面但没有选定会话（例如刚登录），自动创建一个新会话
if st.session_state.get("current_chat_id") is None:
    user_conversations = db.get_user_conversations(current_user["id"])
    
    if user_conversations:
        # 如果有历史会话，默认加载最新的一个
        latest_chat = user_conversations[0]
        st.session_state["current_chat_id"] = latest_chat["id"]
        # 加载该会话的消息
        msgs = db.get_conversation_messages(latest_chat["id"])
        st.session_state["messages"] = msgs if msgs else []
    else:
        # 只有在没有任何会话时，才创建新会话
        new_id = db.create_conversation(current_user["id"], title="New Chat")
        st.session_state["current_chat_id"] = new_id
        st.session_state["messages"] = [{"role": "assistant", "content": "你好！我是你的 AI 助手，全自动分子分析模式已就绪。"}]
        db.add_message(new_id, "assistant", "你好！我是你的 AI 助手，全自动分子分析模式已就绪。")

# --- 初始化本地 LLM ---
if not api_key:
    st.warning("⚠️ Google API Key 无效。")
    st.stop()

try:
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=True, 
    )
except Exception as e:
    st.error(f"模型连接失败: {e}")
    st.stop()

def validate_path_exists(path: str, description: str):
    """检查路径是否存在，不存在则终止"""
    if not path or not os.path.exists(path):
        st.error(f"⛔️ **错误：终止执行**\n\n找不到{description}：`{path}`\n\n请检查文件路径是否正确。")
        st.stop()
    return True

def get_user_workspace():
    """
    根据 session_state 中的用户信息和当前会话ID生成路径
    结构: users/{username}/output/{chat_id}
    """
    if "user" in st.session_state and st.session_state["user"]:
        username = st.session_state["user"]["username"]
        # 确保用户名安全
        safe_username = "".join([c for c in username if c.isalnum() or c in ('-','_')])
        
        # 获取当前会话 ID，如果没有则使用 'temp'
        chat_id = st.session_state.get("current_chat_id", "temp")
        
        # 路径结构: users/hayes/output/123
        workspace = os.path.join("users", safe_username, "output", str(chat_id))
    else:
        # Fallback
        workspace = os.path.join("users", "guest", "output", "temp")
    
    if not os.path.exists(workspace):
        os.makedirs(workspace, exist_ok=True)
    return workspace

def tool_search_db(query_name: str, mol_type: str):
    """Step 1: Search molecule in local database."""
    user_ws = get_user_workspace()
    # 创建一个 search_results 文件夹
    search_dir = os.path.join(user_ws, "search_cache")
    return search_molecule_in_db(query_name, mol_type, search_dir)

def tool_build_optimize(ion_name: str, solvents_json: str, anions_json: str):
    """Step 2: Build and Optimize Cluster."""
    # solvents_json 格式: '[{"name": "DME", "path": "users/.../found_DME.db", "count": 3}]'
    # 或者 '[{"smiles": "COC", "count": 3}]'
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
    """Step 3: Run Inference Pipeline."""
    if model_path in ["None", "", None]:
        model_path = DEFAULT_MODEL_PATH
    
    validate_path_exists(optimized_db_path, "Optimized DB")
    
    db_dir = os.path.dirname(optimized_db_path) # A: final_optimized, B: task_xxx
    parent_dir = os.path.dirname(db_dir)        # A: task_xxx,       B: users/.../output
    
    if os.path.basename(db_dir) == "final_optimized":
        task_root = parent_dir
    elif os.path.basename(db_dir).startswith("task_"): # 简单的启发式判断
        task_root = db_dir
    else:
        # 如果无法确定，为了安全起见，就用 db 所在的目录作为根目录
        # 这样至少文件在一起，不会乱跑
        task_root = db_dir

    # 输出目录: task_root/inference_results
    infer_out = os.path.join(task_root, "inference_results")
    
    result_json_str = run_dm_infer_pipeline(optimized_db_path, model_path, infer_out)
    
    # 打包
    try:
        res_dict = json.loads(result_json_str)
        if res_dict.get("success"):
            csv_path = res_dict.get("csv_path")
            # 优先使用返回的 output_dir，如果没有则使用我们自己定义的 infer_out
            output_dir = res_dict.get("output_dir", infer_out)
            
            # 压缩 output_dir (包含 csv 和 results 文件夹)
            # zip 文件放在 task_root 下，命名为 analysis_package.zip
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

# --- 初始化 Agent ---

custom_system_prefix = """
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

prompt = ChatPromptTemplate.from_messages([
    ("system", custom_system_prefix),
    MessagesPlaceholder(variable_name="chat_history"), # 插入历史记录
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)

# --- 聊天区域显示 ---
st.title("🧪 EMolAgent")

for idx, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        content = msg["content"]
        
        # 解析下载标记
        download_match = re.search(r"\[\[DOWNLOAD:(.*?)\]\]", content)
        # 将标记从显示文本中移除，保持界面整洁
        display_text = re.sub(r"\[\[DOWNLOAD:.*?\]\]", "", content).strip()
        
        st.write(display_text)
        
        # 如果存在文件标记且文件存在，显示下载按钮
        if download_match:
            file_path = download_match.group(1)
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    st.download_button(
                        label="📦 下载分析结果压缩包 (.zip)",
                        data=f,
                        file_name=os.path.basename(file_path),
                        mime="application/zip",
                        key=f"history_btn_{idx}"  # 必须设置唯一的 key
                    )

# --- 处理用户输入 ---
if prompt_input := st.chat_input("请输入指令..."):
    # 1. 立即显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    st.chat_message("user").write(prompt_input)
    
    # 2. 保存用户消息到数据库
    current_chat_id = st.session_state["current_chat_id"]
    db.add_message(current_chat_id, "user", prompt_input)
    
    # 如果是第一条消息（除了欢迎语），更新会话标题
    if len(st.session_state.messages) <= 2: 
        db.update_conversation_title(current_chat_id, prompt_input[:20]) # 截取前20字做标题

    # 3. 准备历史记录传给 Agent (构建 LangChain Message 对象列表)
    # 过滤掉系统欢迎语，只保留稍微近期的对话，或者全部保留
    history_langchain = []
    for m in st.session_state["messages"][:-1]: # 不包含刚发的这条，因为 {input} 里会有
        if m["role"] == "user":
            history_langchain.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history_langchain.append(AIMessage(content=m["content"]))

    # 4. Agent 执行
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        try:
            # 传入 chat_history
            response = agent_executor.invoke(
                {
                    "input": prompt_input,
                    "chat_history": history_langchain 
                }, 
                config={"callbacks": [st_callback]}
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
            
            # 5. 保存 AI 回复
            st.session_state.messages.append({"role": "assistant", "content": output_text})
            db.add_message(current_chat_id, "assistant", output_text)
            
        except Exception as e:
            error_msg = f"执行出错: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            db.add_message(current_chat_id, "assistant", error_msg)
