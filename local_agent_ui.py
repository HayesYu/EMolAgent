import os
import json
# 强制让 Python 在访问本地服务时不使用代理
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

# 导入你的工具库
from tools_lib import run_dptb_inference, update_db_metadata, generate_viz_report

# --- 1. 页面配置 ---
st.set_page_config(page_title="EMol-Vis Local Agent", page_icon="🧪", layout="wide")
st.title("🧪 EMolAgent (Powered by Ollama)")

# --- 侧边栏与控制 ---
with st.sidebar:
    st.header("控制面板")
    
    # 重置按钮
    if st.button("🔄 重置会话 / 停止新任务", type="primary", use_container_width=True):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "你好！我是你的本地 AI 助手。全自动模式已启动，随时待命！"}
        ]
        st.rerun()

    st.markdown("---")
    st.header("模型设置")
    model_name = st.selectbox("选择模型", ["llama3.1", "qwen2.5:7b", "mistral"], index=0)
    temperature = st.slider("温度 (Temperature)", 0.0, 1.0, 0.0)

# --- 2. 初始化本地 LLM ---
llm = ChatOllama(
    model=model_name,
    temperature=temperature,
    base_url="http://localhost:11434"
)

# --- 3. 定义增强版工具 (解决“不知道结果”的问题) ---

def validate_path_exists(path: str, description: str):
    """检查路径是否存在，不存在则终止"""
    if not path or not os.path.exists(path):
        st.error(f"⛔️ **错误：终止执行**\n\n找不到{description}：`{path}`\n\n请检查文件路径是否正确。")
        st.stop()
    return True

def run_dptb_inference_safe(data_root, model_path, output_dir="output", db_name="dump.db"):
    validate_path_exists(data_root, "数据文件夹")
    validate_path_exists(model_path, "模型文件")
    return run_dptb_inference(data_root, model_path, output_dir, db_name)

def update_db_metadata_safe(input_db, input_paths_file, output_db="updated.db"):
    validate_path_exists(input_db, "输入数据库")
    validate_path_exists(input_paths_file, "路径文件")
    return update_db_metadata(input_db, input_paths_file, output_db)

def generate_viz_report_smart(abs_ase_path, npy_folder_path):
    """
    增强版：生成报告后，自动读取 json 内容返回给 Agent。
    这样 Agent 就不需要“再次查找”，也不容易产生幻觉。
    """
    validate_path_exists(abs_ase_path, "ASE数据库")
    validate_path_exists(npy_folder_path, "NPY文件夹")
    
    # 1. 执行原有的生成逻辑
    result_str = generate_viz_report(abs_ase_path, npy_folder_path)
    
    # 2. 自动尝试读取生成的 test_results.json
    json_path = "test_results.json" # 这是 tools_lib 里写死的路径
    json_content = ""
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 将 JSON 转换为字符串，限制长度防止 token 溢出
                json_content = json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            json_content = f"(读取JSON失败: {str(e)})"
    else:
        json_content = "(未找到生成的 test_results.json 文件)"

    # 3. 构造返回给 Agent 的终极信息
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
        description="Step 1. Run inference. Args: data_root, model_path."
    ),
    StructuredTool.from_function(
        func=update_db_metadata_safe,
        name="Update_Metadata",
        description="Step 2. Update metadata. Args: input_db, input_paths_file."
    ),
    StructuredTool.from_function(
        func=generate_viz_report_smart, # 使用增强版
        name="Generate_Visualization",
        description="Step 3. Generate HTML report. Args: abs_ase_path, npy_folder_path."
    )
]

# --- 4. 初始化 Agent ---

# 强化 Prompt，防止死循环
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
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10, # 【防止死循环】限制最大步骤数为10
        early_stopping_method="generate",
        agent_kwargs={
            "prefix": custom_system_prefix,
        }
    )

except Exception as e:
    st.error(f"Agent 初始化失败: {repr(e)}")
    st.stop()

# --- 5. 聊天逻辑 ---

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好！请告诉我数据路径、模型路径和路径文件位置。"}
    ]

# 显示历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 处理输入
if prompt_input := st.chat_input("请输入指令..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    st.chat_message("user").write(prompt_input)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        try:
            # 使用 invoke 接口（LangChain 新版推荐）
            response = agent_executor.run(
                prompt_input, 
                callbacks=[st_callback]
            )
            st.write(response) # 显示最终回答
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            # 捕获可能的错误（如达到最大迭代次数）
            error_msg = f"执行中断或出错: {str(e)}"
            if "Agent stopped due to iteration limit" in str(e):
                error_msg = "⚠️ 任务因步骤过多已强制停止（防止死循环）。请检查上方日志看是否已完成关键步骤。"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})