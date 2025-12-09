import os
from langchain.agents import initialize_agent, AgentType
# [修改点 1] 更改导入库，从 langchain_openai 改为 langchain_ollama
from langchain_ollama import ChatOllama
from langchain.tools import StructuredTool
from tools_lib import run_dptb_inference, update_db_metadata, generate_viz_report

# 1. 配置 LLM
# [修改点 2] 初始化 ChatOllama 而不是 ChatOpenAI
# base_url 默认是 http://localhost:11434，在 WSL 内部运行通常无需修改
llm = ChatOllama(
    model="llama3.1",  # 确保这里是你 ollama pull 下来的模型名字
    temperature=0,
    base_url="http://localhost:11434"
)

# 2. 定义工具 (Tools) - 这部分保持不变
tools = [
    StructuredTool.from_function(
        func=run_dptb_inference,
        name="Run_Inference",
        description="第一步使用。运行深度学习模型推理。输入参数：data_root (数据文件夹), model_path (模型文件路径)。可选：output_dir, db_name。"
    ),
    StructuredTool.from_function(
        func=update_db_metadata,
        name="Update_Metadata",
        description="第二步使用。在推理完成后，修正数据库的 spin 和 charge。输入参数：input_db (上一步生成的db), input_paths_file (包含路径信息的txt文件), output_db (新db名称)。"
    ),
    StructuredTool.from_function(
        func=generate_viz_report,
        name="Generate_Visualization",
        description="第三步使用。生成 MAE 误差报告、Cube 文件和 HTML 可视化。输入参数：abs_ase_path (更新后的db路径), npy_folder_path (第一步生成的output目录)。"
    )
]

# 3. 初始化 Agent
# Llama 3.1 配合 STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION 效果不错
# 我们通过 agent_kwargs 将 system_message 注入到 Agent 的系统提示词中，这样比直接拼接到用户输入更稳定
system_message = """
你是一个计算化学专家 Agent。你的任务是帮助用户完成 DeepPTB 模型的推理和分析流程。
请遵循以下步骤：
1. 运行推理 (Run_Inference)。如果报错 "No trajectory folders found"，可能是 prefix 参数问题。
2. 更新元数据 (Update_Metadata)。这对于后续分析至关重要。
3. 生成可视化报告 (Generate_Visualization)。
4. 最后，读取生成的 json 报告，简要总结 MAE 误差情况反馈给用户。

请严格按顺序调用工具。如果工具报错，请分析错误原因并告知用户，或者尝试修复参数。
"""

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True, # 打印思考过程
    handle_parsing_errors=True, # 容错处理，这对本地模型非常重要！
    agent_kwargs={
        "system_message": system_message # [修改点 3] 推荐将人设放入这里
    }
)

# 5. 交互循环
if __name__ == "__main__":
    print(f"EMol-Vis Agent (Ollama: {llm.model}) 已启动。请输入指令 (输入 'exit' 退出)...")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        try:
            # [修改点 4] 现在可以直接传入用户输入，因为 system_message 已经在 agent 初始化时注入了
            response = agent.run(user_input)
            print(f"\nAgent: {response}")
        except Exception as e:
            print(f"Agent Error: {e}")
