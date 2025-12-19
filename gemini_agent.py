import os
import json

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import StructuredTool
from langchain.callbacks import StreamlitCallbackHandler
import extra_streamlit_components as stx

from tools_lib import run_dptb_inference, generate_viz_report
import database as db

os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

DEFAULT_MODEL_PATH = "/home/hayes/EMolAgent_demo/nnenv.iter147201.pth"

# --- 页面配置 ---
st.set_page_config(page_title="EMolAgent", page_icon="🧪", layout="wide")

def get_manager():
    return stx.CookieManager()

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
                    cookie_manager.set("auth_token", token)
                    st.success("登录成功！")
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
        st.session_state["messages"] = [{"role": "assistant", "content": "你好！我是你的 AI 助手。全自动模式已启动，随时待命！"}]
        # 存入初始欢迎语到数据库
        db.add_message(new_id, "assistant", "你好！我是你的 AI 助手。全自动模式已启动，随时待命！")
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
        st.session_state["messages"] = [{"role": "assistant", "content": "你好！我是你的 AI 助手。全自动模式已启动，随时待命！"}]
        db.add_message(new_id, "assistant", "你好！我是你的 AI 助手。全自动模式已启动，随时待命！")

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

def run_dptb_inference_safe(data_root, model_path=None):
    if model_path in ["None", "null", "", None]:
        model_path = DEFAULT_MODEL_PATH
        st.toast(f"ℹ️ 已自动加载默认模型: {os.path.basename(model_path)}")
    validate_path_exists(data_root, "数据文件夹")
    validate_path_exists(model_path, "模型文件")
    # 自动获取用户目录
    user_ws = get_user_workspace()
    # 定义输出子目录，保持整洁
    output_dir = os.path.join(user_ws, "inference_npy")
    db_name = os.path.join(user_ws, "dump.db") # 数据库也存放在用户目录
    return run_dptb_inference(data_root, model_path, output_dir=output_dir, db_name=db_name)

# def update_db_metadata_safe(input_db, input_paths_file, output_db="updated.db"):
#     validate_path_exists(input_db, "输入数据库")
#     validate_path_exists(input_paths_file, "路径文件")
#     return update_db_metadata(input_db, input_paths_file, output_db)

def generate_viz_report_smart(abs_ase_path, npy_folder_path):
    validate_path_exists(abs_ase_path, "ASE数据库")
    validate_path_exists(npy_folder_path, "NPY文件夹")
    
    # 1. 获取用户目录
    user_ws = get_user_workspace()
    
    # 2. 调用修改后的 tools_lib 函数，传入 work_dir
    result_str = generate_viz_report(abs_ase_path, npy_folder_path, work_dir=user_ws)
    
    # 3. 从用户的目录下读取生成的 JSON
    json_path = os.path.join(user_ws, "test_results.json")
    json_content = ""
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_content = json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            json_content = f"(读取JSON失败: {str(e)})"
    else:
        json_content = "(未找到生成的 test_results.json 文件)"

    final_observation = (
        f"{result_str}\n"
        f"--------------------------------------------------\n"
        f"【重要提示】以下是生成的 JSON 报告内容，请直接通过 Analysis 总结此数据，并给出生成的 cube 文件的路径, 告诉用户内含html, 可查看具体图像然后结束对话：\n"
        f"{json_content}"
    )
    return final_observation

tools = [
    StructuredTool.from_function(
        func=run_dptb_inference_safe,
        name="Run_Inference",
        description="Step 1. Run inference. Args: data_root. (Output will be saved to your user workspace)"
    ),
    # StructuredTool.from_function(
    #     func=update_db_metadata_safe,
    #     name="Update_Metadata",
    #     description="Step 2. Update metadata. Args: input_db, input_paths_file."
    # ),
    StructuredTool.from_function(
        func=generate_viz_report_smart,
        name="Generate_Visualization",
        description="Step 2. Generate HTML report. Args: abs_ase_path, npy_folder_path."
    )
]

# --- 初始化 Agent (使用新版 Tool Calling API) ---

custom_system_prefix = """
你是一个计算化学 AI 助手。请按顺序执行以下步骤：
1. Run_Inference (执行推理，生成 npy 和 db 文件)
2. Generate_Visualization (计算误差并生成可视化文件)
3. 请根据返回的 JSON 数据回答用户的误差结果，并给出生成的 cube 文件的路径，告诉用户内含 html 文件, 可查看具体图像, 然后结束对话。
【极重要规则】：
- 当你执行完 "Generate_Visualization" 后，工具会直接返回 JSON 数据内容。
- **一旦你看到了 JSON 数据，必须立即停止调用任何工具！**
- **绝对禁止**再次调用 Run_Inference。
- 请直接根据返回的 JSON 数据回答用户的误差结果，并给出生成的 cube 文件的路径, 告诉用户内含html, 可查看具体图像, 然后结束对话。
- 你的最后一句必须是："任务已完成。"
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

# 显示当前会话的消息
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

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
            
            st.write(output_text)
            
            # 5. 保存 AI 回复
            st.session_state.messages.append({"role": "assistant", "content": output_text})
            db.add_message(current_chat_id, "assistant", output_text)
            
        except Exception as e:
            error_msg = f"执行出错: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            db.add_message(current_chat_id, "assistant", error_msg)