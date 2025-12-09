import os
# 强制让 Python 在访问本地服务时不使用代理
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
# 为了保险，也可以把下面这两行加上，彻底屏蔽脚本的代理设置
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
import streamlit as st
from langchain_community.chat_models import ChatOllama
# [关键修改] 改回使用 initialize_agent，它能自动处理 Prompt 结构，不再手动拼接
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain.callbacks import StreamlitCallbackHandler

# 导入你的工具库
from tools_lib import run_dptb_inference, update_db_metadata, generate_viz_report

# --- 1. 页面配置 ---
st.set_page_config(page_title="EMol-Vis Local Agent", page_icon="🧪")
st.title("🧪 EMolAgent (Powered by Ollama)")

with st.sidebar:
    st.header("模型设置")
    model_name = st.selectbox("选择模型", ["llama3.1", "qwen2.5:7b", "mistral"], index=0)
    temperature = st.slider("温度 (Temperature)", 0.0, 1.0, 0.0)
    st.markdown("---")
    st.markdown("✅ **状态**: 本地运行中 (Classic Mode)")
    st.markdown("🚫 **网络**: 本地")

# --- 2. 初始化本地 LLM ---
llm = ChatOllama(
    model=model_name,
    temperature=temperature,
    base_url="http://localhost:11434"
)

# --- 3. 定义工具 ---
tools = [
    StructuredTool.from_function(
        func=run_dptb_inference,
        name="Run_Inference",
        description="Step 1. Run deep learning inference. Args: data_root, model_path."
    ),
    StructuredTool.from_function(
        func=update_db_metadata,
        name="Update_Metadata",
        description="Step 2. Correct spin/charge metadata. Args: input_db, input_paths_file."
    ),
    StructuredTool.from_function(
        func=generate_viz_report,
        name="Generate_Visualization",
        description="Step 3. Generate HTML/MAE report. Args: abs_ase_path, npy_folder_path."
    )
]

# --- 4. 初始化 Agent (使用 initialize_agent) ---

# 定义你的个性化系统提示词 (System Prompt)
# 我们把它放在 Agent 的 "prefix" 中，这样既保留了你的要求，又不会破坏 Agent 的内部结构
custom_system_prefix = """
你是一个精通 Python 和计算化学的 AI 助手。
你的目标是帮助用户完成 DeepPTB 模型的推理和分析流程。

请严格遵循以下核心规则：
1. **严格顺序**：必须按照 Run_Inference -> Update_Metadata -> Generate_Visualization 的顺序执行。
2. **参数检查**：如果工具报错，请仔细检查错误信息（如文件路径不存在）并尝试修复参数后重试。
3. **最终反馈**：在可视化生成后，明确告诉用户 HTML 文件的保存路径。
"""

try:
    # [核心修改] 使用 initialize_agent 自动组装
    # agent_kwargs 用于注入你的自定义 Prompt
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True, # 自动纠正格式错误
        agent_kwargs={
            "prefix": custom_system_prefix, # 注入你的指令
            # "input_variables": ["input", "agent_scratchpad"] # 让它自动处理
        }
    )

except Exception as e:
    # 使用 repr(e) 打印完整的错误对象，防止错误信息为空
    st.error(f"Agent 初始化失败: {repr(e)}")
    st.stop()

# --- 5. 聊天界面逻辑 ---

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好！我是你的本地 AI 助手。全自动模式已启动，随时待命！"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt_input := st.chat_input("请输入指令..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    st.chat_message("user").write(prompt_input)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        try:
            # initialize_agent 返回的就是 Executor，直接调用 run 或 invoke
            response = agent_executor.run(
                prompt_input, 
                callbacks=[st_callback]
            )
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"执行出错: {repr(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"执行出错: {e}"})