import os
import json

import streamlit as st
# --- 1. 导入必要的库 ---
from langchain_google_genai import ChatGoogleGenerativeAI
# 注意：这里引入了 create_tool_calling_agent 和 AgentExecutor
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from langchain.callbacks import StreamlitCallbackHandler

from tools_lib import run_dptb_inference, update_db_metadata, generate_viz_report

# --- 2. 环境变量配置 ---
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

DEFAULT_MODEL_PATH = "/home/hayes/EMolAgent_demo/nnenv.iter147201.pth"

# --- 3. 页面配置 ---
st.set_page_config(page_title="EMol-Vis Local Agent", page_icon="🧪", layout="wide")
st.title("🧪 EMolAgent")

# --- 4. 侧边栏与控制 ---
with st.sidebar:
    st.header("控制面板")
    
    # 重置按钮
    if st.button("🔄 重置会话 / 停止新任务", type="primary", use_container_width=True):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "你好！我是你的 AI 助手。全自动模式已启动，随时待命！"}
        ]
        st.rerun()

    st.markdown("---")
    st.header("模型设置")
    model_name = st.selectbox(
        "选择模型", 
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-exp"], 
        index=0
    )
    api_key = st.text_input("Google API Key", type="password", help="在此输入你的 Gemini API Key")
    temperature = st.slider("温度 (Temperature)", 0.0, 1.0, 0.0)
    st.info(f"**当前默认模型**:\n{os.path.basename(DEFAULT_MODEL_PATH)}")

# --- 5. 初始化本地 LLM ---
if not api_key:
    st.warning("⚠️ 请在左侧侧边栏输入 Google API Key 以启动 Agent。")
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

# --- 6. 定义增强版工具 ---

def validate_path_exists(path: str, description: str):
    """检查路径是否存在，不存在则终止"""
    if not path or not os.path.exists(path):
        st.error(f"⛔️ **错误：终止执行**\n\n找不到{description}：`{path}`\n\n请检查文件路径是否正确。")
        st.stop()
    return True

def run_dptb_inference_safe(data_root, model_path=None, output_dir="output", db_name="dump.db"):
    if model_path in ["None", "null", "", None]:
        model_path = DEFAULT_MODEL_PATH
        st.toast(f"ℹ️ 已自动加载默认模型: {os.path.basename(model_path)}")
    validate_path_exists(data_root, "数据文件夹")
    validate_path_exists(model_path, "模型文件")
    return run_dptb_inference(data_root, model_path, output_dir, db_name)

def update_db_metadata_safe(input_db, input_paths_file, output_db="updated.db"):
    validate_path_exists(input_db, "输入数据库")
    validate_path_exists(input_paths_file, "路径文件")
    return update_db_metadata(input_db, input_paths_file, output_db)

def generate_viz_report_smart(abs_ase_path, npy_folder_path):
    validate_path_exists(abs_ase_path, "ASE数据库")
    validate_path_exists(npy_folder_path, "NPY文件夹")
    
    result_str = generate_viz_report(abs_ase_path, npy_folder_path)
    json_path = "test_results.json" 
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
        description="Step 1. Run inference. Args: data_root. (Model path defaults to built-in if not provided)"
    ),
    StructuredTool.from_function(
        func=update_db_metadata_safe,
        name="Update_Metadata",
        description="Step 2. Update metadata. Args: input_db, input_paths_file."
    ),
    StructuredTool.from_function(
        func=generate_viz_report_smart,
        name="Generate_Visualization",
        description="Step 3. Generate HTML report. Args: abs_ase_path, npy_folder_path."
    )
]

# --- 7. 初始化 Agent (使用新版 Tool Calling API) ---

custom_system_prefix = """
你是一个计算化学 AI 助手。请按顺序执行以下步骤：
1. Run_Inference
2. Update_Metadata
3. Generate_Visualization
4. 请根据返回的 JSON 数据回答用户的误差结果，并给出生成的 cube 文件的路径，告诉用户内含 html 文件, 可查看具体图像, 然后结束对话。
【极重要规则】：
- 当你执行完 "Generate_Visualization" 后，工具会直接返回 JSON 数据内容。
- **一旦你看到了 JSON 数据，必须立即停止调用任何工具！**
- **绝对禁止**再次调用 Run_Inference。
- 请直接根据返回的 JSON 数据回答用户的误差结果，并给出生成的 cube 文件的路径, 告诉用户内含html, 可查看具体图像, 然后结束对话。
- 你的最后一句必须是："任务已完成。"
"""

try:
    # 1. 创建 Prompt (显式包含 system 和 placeholder)
    prompt = ChatPromptTemplate.from_messages([
        ("system", custom_system_prefix),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 2. 创建 Agent (针对 Gemini/OpenAI 等支持 Function Calling 的模型)
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 3. 创建执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10, 
    )

except Exception as e:
    st.error(f"Agent 初始化失败: {repr(e)}")
    st.stop()

# --- 8. 聊天逻辑 ---

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好！请告诉我数据路径、和 Spin/Charge 映射文件位置。"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt_input := st.chat_input("请输入指令..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    st.chat_message("user").write(prompt_input)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        try:
            # --- 关键修改：使用 .invoke 而不是 .run ---
            response = agent_executor.invoke(
                {"input": prompt_input}, 
                config={"callbacks": [st_callback]}
            )
            # invoke 返回的是一个字典，结果在 'output' 键中
            output_text = response["output"]
            
            st.write(output_text) 
            st.session_state.messages.append({"role": "assistant", "content": output_text})
        except Exception as e:
            error_msg = f"执行中断或出错: {str(e)}"
            if "Agent stopped due to iteration limit" in str(e):
                error_msg = "⚠️ 任务因步骤过多已强制停止（防止死循环）。"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})