"""
EMolAgent Streamlit 主应用

提供基于 Web 的用户界面，集成 LangChain Agent 进行分子计算和知识问答。
"""

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

from emolagent.utils.logger import logger
from emolagent.utils.paths import get_resource_path, get_project_root

from emolagent.database import db
from emolagent.core.tools import (
    search_molecule_in_db,
    build_and_optimize_cluster,
    build_multiple_clusters,
    run_dm_infer_pipeline,
    compress_directory,
    get_task_queue_status,
)

from emolagent.knowledge import (
    search_knowledge,
    build_index,
    get_index_stats,
    LITERATURE_PATH,
)

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

from emolagent.visualization import (
    create_structure_preview_html, 
    load_structure_from_db,
    load_all_structures_from_db,
    get_structure_count_from_db,
    create_gaussian_view_style_viewer,
    create_orbital_viewer,
    find_orbital_files,
    find_li_deformation_files,
    create_li_deformation_viewer,
    create_analysis_visualization_html,
    find_esp_files,
    create_esp_viewer,
)
import streamlit.components.v1 as components


DEFAULT_MODEL_PATH = get_resource_path("models", "nnenv.ep154.pth")

ADMIN_USERS = ["hayes"]

WELCOME_MESSAGE = """您好！我是 EMolAgent，您的计算化学 AI 助手。

我具备两大核心能力：

🔬 **分子团簇计算**
从本地数据库检索分子、构建并优化团簇结构，预测 HOMO/LUMO、偶极矩及静电势等电子性质。
示例：「请构建一个包含 1个Li离子、3个DME分子 和 1个FSI阴离子 的团簇」

📚 **文献知识问答**
基于数百篇 AI for Science 和电解液领域文献，回答相关学术问题。
示例：「什么是溶剂化结构？CIP和SSIP有什么区别？」「介绍一下 GNN 在分子性质预测中的应用」

请告诉我您的需求，我将为您提供帮助！"""

CUSTOM_SYSTEM_PREFIX = """
你是一个计算化学 AI 助手 EMolAgent。你有两大核心能力：

## 能力一：分子团簇计算
请遵循以下工作流来处理用户的分子计算请求：

### 重要：识别用户意图
用户意图可以分为以下几类，请仔细判断：

1. **只生成结构（不分析）**：
   - 关键词：「生成结构」「构建团簇」「创建分子」
   - **不包含**：「分析」「电子结构」「预测性质」「HOMO」「LUMO」等
   - 操作：只调用构建工具，**不要调用** `Run_Inference_Pipeline`

2. **生成并分析（完整流程）**：
   - 关键词：「生成并分析」「计算电子结构」「预测性质」「进行分析」「电子结构分析」
   - 操作：调用构建工具 + `Run_Inference_Pipeline`

3. **对已有结构进行分析**：
   - 关键词：「对上面的结构进行分析」「分析刚才的结构」「继续分析」
   - 操作：从对话历史找到 `optimized_db` 路径，只调用 `Run_Inference_Pipeline`

### ⚠️ 关键：单配方 vs 多配方
- **单配方**：用户只描述了一种配方（如"1Li+3DME+1FSI"）
  - 只生成结构：`Build_Structure_Only`
  - 生成并分析：`Build_and_Optimize` + `Run_Inference_Pipeline`

- **多配方**：用户描述了多种不同配方（如"构建A配方...然后构建B配方..."）
  - **必须**使用 `Build_Multiple_Clusters` 一次性处理所有配方
  - 只生成结构：`Build_Multiple_Clusters`（不调用 Run_Inference_Pipeline）
  - 生成并分析：`Build_Multiple_Clusters` + `Run_Inference_Pipeline`

**示例判断**：
- 「构建 1Li+3DME+1FSI 和 1Li+2DME+2FSI 两个团簇」→ 多配方 + 只生成结构 → `Build_Multiple_Clusters`（完毕）
- 「构建 1Li+3DME+1FSI 和 1Li+2DME+2FSI 两个团簇并进行电子结构分析」→ 多配方 + 分析 → `Build_Multiple_Clusters` + `Run_Inference_Pipeline`
- 「构建 1Li+3DME+1FSI」→ 单配方 + 只生成结构 → `Build_Structure_Only`（完毕）

### 工作流步骤：

1.  **解析需求**：
    * 识别中心离子、溶剂、阴离子及其数量
    * 判断是单配方还是多配方
    * **关键判断**：用户是否要求进行电子结构分析

2.  **数据库检索 (Search_Molecule_DB)**：
    * 对每个分子（溶剂或阴离子），先调用 `Search_Molecule_DB` 查库
    * Solvent 查 'solvent' 类型，Anion 查 'anion' 类型
    * 如果找到 `db_path`，告诉用户"已在库中找到 XXX"
    * 如果没找到，使用 SMILES

3.  **建模与优化**：
    * 单配方只生成结构：`Build_Structure_Only`
    * 单配方完整分析：`Build_and_Optimize` → `Run_Inference_Pipeline`
    * 多配方只生成结构：`Build_Multiple_Clusters`（**到此结束，不要调用分析**）
    * 多配方完整分析：`Build_Multiple_Clusters` → `Run_Inference_Pipeline`
    * 构造 JSON 参数时：
      - 有 DB 路径：`{"name": "DME", "path": "...", "count": 3}`
      - 无 DB 路径：`{"smiles": "...", "count": 3}`

4.  **电子结构推断 (Run_Inference_Pipeline)**：
    * **仅当用户明确要求分析时才调用**
    * 使用上一步返回的 `optimized_db` 路径
    * 对于多配方，只需调用一次，会处理所有结构

5.  **最终报告**：
    * 只生成结构：展示 3D 预览，提示用户可后续分析
    * 执行了分析：展示电子性质，保留 `[[DOWNLOAD:...]]` 链接

### 记住：
- 用户没说「分析」「电子结构」→ 不要调用 `Run_Inference_Pipeline`
- 用户后续说「分析上面的结构」→ 从历史找 `optimized_db`，调用 `Run_Inference_Pipeline`
- 多配方 = 使用 `Build_Multiple_Clusters`，不要多次调用单配方工具

## 能力二：文献知识问答 (Search_Knowledge_Base)
当用户询问以下类型的问题时，使用 `Search_Knowledge_Base` 工具：
- AI for Science 相关模型和方法（如 GNN、Transformer、扩散模型等）
- 电解液性质、溶剂化结构、离子传输机理
- 电池材料、锂离子/钠离子电池
- 分子模拟方法、DFT计算、机器学习势函数
- 任何需要文献支撑的科学概念解释

**知识问答工作流**：
1. 理解用户问题的核心概念
2. 调用 `Search_Knowledge_Base` 搜索相关文献
3. 基于检索到的内容，结合你的知识进行综合回答
4. **必须引用来源**，格式如：「根据文献 [xxx.pdf] ...」

【注意】
* 如果用户说"3个DME"，意思是 count=3。
* FSI 通常是阴离子。
* 一步步执行，不要跳过"查库"步骤，因为库内构型质量最高。
* 对于知识性问题，优先使用知识库搜索，确保回答有文献依据。
"""

# --- 页面配置 ---
st.set_page_config(page_title="EMolAgent", page_icon="🧪", layout="wide")


# ==============================================================================
# 1. 辅助函数定义
# ==============================================================================

@st.cache_resource(ttl=86400)
def schedule_cleanup():
    """定时清理任务。"""
    db.cleanup_old_data(days=30)
    return True


def get_manager():
    return stx.CookieManager(key="auth_cookie_manager")


def validate_path_exists(path: str, description: str):
    """检查路径是否存在，不存在则终止。"""
    if not path or not os.path.exists(path):
        st.error(f"⛔️ **错误：终止执行**\n\n找不到{description}：`{path}`\n\n请检查文件路径是否正确。")
        st.stop()
    return True


def get_user_workspace():
    """根据 session_state 中的用户信息和当前会话ID生成路径。"""
    if "user" in st.session_state and st.session_state["user"]:
        username = st.session_state["user"]["username"]
        safe_username = "".join([c for c in username if c.isalnum() or c in ("-", "_")])
        chat_id = st.session_state.get("current_chat_id", "temp")
        workspace = os.path.join(get_project_root(), "users", safe_username, "output", str(chat_id))
    else:
        workspace = os.path.join(get_project_root(), "users", "guest", "output", "temp")

    if not os.path.exists(workspace):
        os.makedirs(workspace, exist_ok=True)
    return workspace


def get_user_workspace_from_ids(username: str | None, chat_id: str | None):
    safe_username = "".join([c for c in (username or "guest") if c.isalnum() or c in ("-", "_")])
    safe_chat_id = str(chat_id or "temp")
    workspace = os.path.join(get_project_root(), "users", safe_username, "output", safe_chat_id)
    os.makedirs(workspace, exist_ok=True)
    return workspace


# ==============================================================================
# 2. Tools
# ==============================================================================

@dataclass
class Context:
    """自定义运行时上下文。"""
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
    """在本地数据库中搜索分子。"""
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
    """构建并优化团簇。"""
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
    "Build_Structure_Only",
    description=(
        "Build and optimize a molecular cluster structure WITHOUT running electronic structure analysis. "
        "Use this when user just wants to generate/build a SINGLE structure. "
        "Args: ion_name (str), solvents_json (JSON list), anions_json (JSON list). "
        "Returns the optimized structure path and a 3D visualization for user confirmation."
    ),
)
def tool_build_structure_only(ion_name: str, solvents_json: str, anions_json: str, runtime: ToolRuntime[Context]) -> str:
    """仅构建结构，不进行电子结构分析。"""
    try:
        solvents = json.loads(solvents_json) if solvents_json else []
        anions = json.loads(anions_json) if anions_json else []
    except Exception:
        return "Error parsing JSON inputs."

    user_ws = get_user_workspace_from_ids(runtime.context.username, runtime.context.chat_id)
    task_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
    task_dir = os.path.join(user_ws, f"task_{task_id}")
    
    result = build_and_optimize_cluster(ion_name, solvents, anions, task_dir)
    
    try:
        res_dict = json.loads(result)
        if res_dict.get("success"):
            optimized_db = res_dict.get("optimized_db")
            return json.dumps({
                "success": True,
                "optimized_db": optimized_db,
                "task_dir": task_dir,
                "msg": f"结构已生成并优化完成。路径: {optimized_db}",
                "visualization_marker": f"[[STRUCTURE_PREVIEW:{optimized_db}]]"
            })
        return result
    except:
        return result


@tool(
    "Build_Multiple_Clusters",
    description=(
        "Build MULTIPLE clusters with DIFFERENT recipes in ONE call. "
        "USE THIS when user requests multiple different cluster configurations in a single request. "
        "For example: 'build 1Li+3DME+1FSI AND 1Li+2DME+2FSI' should use this tool ONCE, not Build_and_Optimize twice. "
        "Args: ion_name (str), recipes_json (JSON list of recipes). "
        "Each recipe: {'solvents': [{'name': 'DME', 'path': '...', 'count': 3}], 'anions': [{'name': 'FSI', 'path': '...', 'count': 1}]}. "
        "Returns the optimized DB path containing ALL clusters."
    ),
)
def tool_build_multiple_clusters(ion_name: str, recipes_json: str, runtime: ToolRuntime[Context]) -> str:
    """批量构建多个不同配方的团簇。"""
    try:
        recipes = json.loads(recipes_json) if recipes_json else []
    except Exception:
        return json.dumps({"success": False, "msg": "Error parsing recipes_json."})

    if not recipes or len(recipes) == 0:
        return json.dumps({"success": False, "msg": "No recipes provided."})

    user_ws = get_user_workspace_from_ids(runtime.context.username, runtime.context.chat_id)
    task_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
    task_dir = os.path.join(user_ws, f"task_{task_id}")
    
    result = build_multiple_clusters(ion_name, recipes, task_dir)
    
    try:
        res_dict = json.loads(result)
        if res_dict.get("success"):
            optimized_db = res_dict.get("optimized_db")
            recipes_count = res_dict.get("recipes_count", len(recipes))
            return json.dumps({
                "success": True,
                "optimized_db": optimized_db,
                "task_dir": task_dir,
                "recipes_count": recipes_count,
                "msg": f"成功构建 {recipes_count} 个配方的团簇。路径: {optimized_db}",
                "visualization_marker": f"[[STRUCTURE_PREVIEW:{optimized_db}]]"
            })
        return result
    except:
        return result


@tool(
    "Run_Inference_Pipeline",
    description=(
        "Run DPTB inference and electronic structure analysis on optimized DB. "
        "Args: optimized_db_path (str), model_path (optional). "
        "Returns a string containing [[DOWNLOAD:...]] zip link on success."
    ),
)
def tool_infer_pipeline(optimized_db_path: str, model_path: str | None, runtime: ToolRuntime[Context]) -> str:
    """运行电子结构推断。"""
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
    
    # 获取用户 ID 用于日志追踪
    user_id = None
    if runtime and runtime.context:
        user_id = runtime.context.username or runtime.context.user_id
    
    result_json_str = run_dm_infer_pipeline(
        optimized_db_path, 
        model_path, 
        infer_out,
        user_id=user_id
    )

    try:
        res_dict = json.loads(result_json_str)
        if res_dict.get("success"):
            csv_path = res_dict.get("csv_path")
            output_dir = res_dict.get("output_dir", infer_out)
            gpu_id = res_dict.get("gpu_id", "N/A")
            zip_base_name = os.path.join(task_root, f"analysis_package_{run_id}")
            zip_path = compress_directory(output_dir, zip_base_name)

            return (
                f"推理完成 (GPU {gpu_id})。\n"
                f"CSV摘要路径: {csv_path}\n"
                f"数据预览: {res_dict.get('data_preview')}\n"
                f"[[ANALYSIS_VISUALIZATION:{optimized_db_path}|{infer_out}]]\n"
                f"[[DOWNLOAD:{zip_path}]]"
            )
        return f"推理出错: {result_json_str}"
    except Exception as e:
        return f"Error processing inference results: {e}"


@tool(
    "Search_Knowledge_Base",
    description=(
        "Search the literature knowledge base for AI4Science and electrolyte-related content. "
        "Use this tool when user asks about: AI models, machine learning methods, electrolyte properties, "
        "battery materials, molecular simulation theories, or any scientific concepts. "
        "Args: query (str) - the search query in natural language, top_k (int, optional) - number of results. "
        "Returns relevant excerpts from academic papers with source citations."
    ),
)
def tool_search_knowledge(query: str, top_k: int = 5) -> str:
    """搜索知识库。"""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return "Error: Google API Key not configured."
    
    try:
        results = search_knowledge(query, api_key, top_k=top_k)
        
        if not results:
            return "未找到相关文献内容。请尝试换一种表达方式或更具体的关键词。"
        
        output_parts = [f"找到 {len(results)} 条相关文献内容：\n"]
        
        for i, r in enumerate(results, 1):
            output_parts.append(
                f"**[{i}] {r['source']}** (相关度: {r['relevance_score']:.2f})\n"
                f"分类: {r['category'] or '根目录'}\n"
                f"内容摘要:\n> {r['content'][:500]}{'...' if len(r['content']) > 500 else ''}\n"
            )
        
        return "\n---\n".join(output_parts)
    
    except Exception as e:
        return f"知识库搜索出错: {str(e)}"


TOOLS = [tool_search_db, tool_build_structure_only, tool_build_multiple_clusters, tool_build_optimize, tool_infer_pipeline, tool_search_knowledge]


# ==============================================================================
# 3. Agent 初始化
# ==============================================================================

@dataclass
class ResponseFormat:
    """结构化响应模式。"""
    output: str


@st.cache_resource(show_spinner=False)
def get_checkpointer() -> InMemorySaver:
    return InMemorySaver()


def build_agent(model_name: str, temperature: float, api_key: str):
    """构建并返回 LangChain agent。"""
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
        checkpointer=checkpointer,
    )
    return agent


# ==============================================================================
# 4. UI：登录 / 主界面
# ==============================================================================

def login_ui(cookie_manager):
    """处理登录和注册的 UI 渲染。"""
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


def _show_esp_info(info_path: str):
    """显示 ESP 极值信息的辅助函数。"""
    try:
        import json as json_module
        with open(info_path, 'r') as f:
            esp_info = json_module.load(f)
        
        st.markdown("---")
        st.markdown("**ESP 极值信息**")
        col_max, col_min = st.columns(2)
        with col_max:
            max_val = esp_info.get('ESP_max_eV', 'N/A')
            max_loc = esp_info.get('ESP_max_location_Ang', [])
            st.metric("最大值 (eV)", f"{max_val:.4f}" if isinstance(max_val, (int, float)) else max_val)
            if max_loc:
                st.caption(f"位置: ({max_loc[0]:.2f}, {max_loc[1]:.2f}, {max_loc[2]:.2f}) Å")
        with col_min:
            min_val = esp_info.get('ESP_min_eV', 'N/A')
            min_loc = esp_info.get('ESP_min_location_Ang', [])
            st.metric("最小值 (eV)", f"{min_val:.4f}" if isinstance(min_val, (int, float)) else min_val)
            if min_loc:
                st.caption(f"位置: ({min_loc[0]:.2f}, {min_loc[1]:.2f}, {min_loc[2]:.2f}) Å")
    except Exception:
        pass


def render_message_with_download(role: str, content: Any, key_prefix: str):
    """将特殊标记渲染为可交互组件。"""
    text = normalize_chat_content(content)

    with st.chat_message(role):
        structure_match = re.search(r"\[\[STRUCTURE_PREVIEW:(.*?)\]\]", text)
        analysis_match = re.search(r"\[\[ANALYSIS_VISUALIZATION:(.*?)\|(.*?)\]\]", text)
        download_match = re.search(r"\[\[DOWNLOAD:(.*?)\]\]", text)
        
        display_text = re.sub(r"\[\[STRUCTURE_PREVIEW:.*?\]\]", "", text)
        display_text = re.sub(r"\[\[ANALYSIS_VISUALIZATION:.*?\]\]", "", display_text)
        display_text = re.sub(r"\[\[DOWNLOAD:.*?\]\]", "", display_text).strip()
        st.write(display_text)

        if analysis_match:
            db_path = analysis_match.group(1).strip()
            infer_dir = analysis_match.group(2).strip()
            
            st.markdown("### 🔬 分析结果可视化")
            
            # 查找 Li deformation 文件和 ESP 文件
            li_deform_files = find_li_deformation_files(infer_dir)
            esp_files_list = find_esp_files(infer_dir)
            has_esp = len(esp_files_list) > 0
            
            # 查找轨道文件
            orbital_files = find_orbital_files(infer_dir)
            has_homo = len(orbital_files.get('homo', [])) > 0
            has_lumo = len(orbital_files.get('lumo', [])) > 0
            
            # 根据可用文件决定 tab 数量
            tab_names = ["🧬 团簇结构"]
            if has_homo:
                tab_names.append("🔵 HOMO 轨道")
            if has_lumo:
                tab_names.append("🟢 LUMO 轨道")
            if has_esp:
                tab_names.append("⚡ 静电势 (ESP)")
            if li_deform_files:
                tab_names.append("💠 Li Deformation")
            
            tabs = st.tabs(tab_names)
            tab_idx = 0
            tab_structure = tabs[tab_idx]; tab_idx += 1
            tab_homo = tabs[tab_idx] if has_homo else None; tab_idx += (1 if has_homo else 0)
            tab_lumo = tabs[tab_idx] if has_lumo else None; tab_idx += (1 if has_lumo else 0)
            tab_esp = tabs[tab_idx] if has_esp else None; tab_idx += (1 if has_esp else 0)
            tab_li_deform = tabs[tab_idx] if li_deform_files else None
            
            # 预加载结构信息，用于生成一致的标签
            structure_labels = {}  # id -> label 映射
            structures_data = []   # 保存结构数据供复用
            if os.path.exists(db_path):
                try:
                    structures_data = load_all_structures_from_db(db_path, max_count=20)
                    for i, (atoms, meta) in enumerate(structures_data):
                        solv_name = meta.get('solvent_name', '')
                        anion_name = meta.get('anion_name', '')
                        n_solv = meta.get('n_solv', 0)
                        n_anion = meta.get('n_anion', 0)
                        category = meta.get('category', '')
                        
                        if anion_name and n_anion > 0:
                            label = f"{n_solv}{solv_name}+{n_anion}{anion_name}"
                        else:
                            label = f"{n_solv}{solv_name}"
                        if category:
                            label = f"[{category}] {label}"
                        structure_labels[str(i)] = label
                except Exception:
                    pass
            
            def get_structure_label(file_id: str, index: int) -> str:
                """根据文件 ID 获取结构标签。"""
                if file_id in structure_labels:
                    return structure_labels[file_id]
                # 尝试用 index 查找
                if str(index) in structure_labels:
                    return structure_labels[str(index)]
                return f"结构{file_id}"
            
            with tab_structure:
                if os.path.exists(db_path):
                    try:
                        total_count = len(structures_data) if structures_data else get_structure_count_from_db(db_path)
                        max_display = 3  # 最多显示 3 个结构
                        
                        if total_count <= 1:
                            # 单个结构
                            if structures_data:
                                atoms, meta = structures_data[0]
                            else:
                                atoms = load_structure_from_db(db_path)
                                meta = {}
                            if atoms:
                                viewer_html = create_gaussian_view_style_viewer(
                                    atoms,
                                    width=650,
                                    height=500,
                                    style="sphere+stick",
                                    add_lighting=True
                                )
                                components.html(viewer_html, height=560, scrolling=False)
                                st.caption(f"化学式: {atoms.get_chemical_formula()} | 原子数: {len(atoms)}")
                            else:
                                st.warning("无法加载结构预览")
                        else:
                            # 多个结构
                            displayed_count = min(total_count, max_display)
                            st.markdown(f"**共 {total_count} 个团簇，显示前 {displayed_count} 个**")
                            
                            structures_to_show = structures_data[:max_display]
                            
                            if structures_to_show:
                                # 生成子 tab 名称，复用 structure_labels
                                sub_tab_names = [f"#{i+1}: {structure_labels.get(str(i), f'结构{i}')}" 
                                                 for i in range(len(structures_to_show))]
                                
                                sub_tabs = st.tabs(sub_tab_names)
                                
                                for i, (sub_tab, (atoms, meta)) in enumerate(zip(sub_tabs, structures_to_show)):
                                    with sub_tab:
                                        viewer_html = create_gaussian_view_style_viewer(
                                            atoms,
                                            width=650,
                                            height=450,
                                            style="sphere+stick",
                                            add_lighting=True
                                        )
                                        components.html(viewer_html, height=510, scrolling=False)
                                        
                                        # 显示配方信息
                                        solv_name = meta.get('solvent_name', 'Unknown')
                                        anion_name = meta.get('anion_name', '')
                                        n_solv = meta.get('n_solv', 0)
                                        n_anion = meta.get('n_anion', 0)
                                        ion = meta.get('ion', 'Li')
                                        
                                        formula_parts = [f"1x{ion}⁺"]
                                        if n_solv > 0:
                                            formula_parts.append(f"{n_solv}x{solv_name}")
                                        if n_anion > 0 and anion_name:
                                            formula_parts.append(f"{n_anion}x{anion_name}⁻")
                                        
                                        st.caption(f"配方: {' + '.join(formula_parts)} | 化学式: {atoms.get_chemical_formula()} | 原子数: {len(atoms)}")
                                
                                if total_count > max_display:
                                    st.info(f"💡 还有 {total_count - max_display} 个团簇未显示。完整结果请下载分析包查看。")
                            else:
                                st.warning("无法加载结构预览")
                    except Exception as e:
                        st.error(f"结构预览失败: {e}")
                else:
                    st.warning(f"结构文件不存在: {db_path}")
            
            # HOMO Tab
            if tab_homo is not None and has_homo:
                with tab_homo:
                    homo_files = orbital_files.get('homo', [])
                    try:
                        # 等值面设置
                        st.markdown("**等值面设置**")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            homo_iso = st.slider(
                                "等值面大小",
                                min_value=0.005,
                                max_value=0.1,
                                value=0.02,
                                step=0.005,
                                format="%.3f",
                                key=f"{key_prefix}_homo_iso",
                                help="调大：轨道包络面收缩；调小：轨道包络面扩展"
                            )
                        with col2:
                            st.metric("当前值", f"{homo_iso:.3f}")
                        
                        if len(homo_files) == 1:
                            # 单个文件
                            homo_html = create_orbital_viewer(
                                homo_files[0]['path'],
                                width=650,
                                height=500,
                                iso_value=homo_iso,
                                orbital_type="HOMO"
                            )
                            components.html(homo_html, height=560, scrolling=False)
                            st.caption(f"文件: {os.path.basename(homo_files[0]['path'])}")
                        else:
                            # 多个文件，使用 tabs 切换
                            st.markdown(f"**共 {len(homo_files)} 个 HOMO 轨道文件**")
                            homo_tab_names = [f"#{i+1}: {get_structure_label(f['id'], i)}" for i, f in enumerate(homo_files)]
                            homo_sub_tabs = st.tabs(homo_tab_names)
                            
                            for i, (sub_tab, homo_file) in enumerate(zip(homo_sub_tabs, homo_files)):
                                with sub_tab:
                                    homo_html = create_orbital_viewer(
                                        homo_file['path'],
                                        width=650,
                                        height=450,
                                        iso_value=homo_iso,
                                        orbital_type="HOMO"
                                    )
                                    components.html(homo_html, height=510, scrolling=False)
                                    st.caption(f"文件: {os.path.basename(homo_file['path'])}")
                    except Exception as e:
                        st.error(f"HOMO 可视化失败: {e}")
            
            # LUMO Tab
            if tab_lumo is not None and has_lumo:
                with tab_lumo:
                    lumo_files = orbital_files.get('lumo', [])
                    try:
                        # 等值面设置
                        st.markdown("**等值面设置**")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            lumo_iso = st.slider(
                                "等值面大小",
                                min_value=0.005,
                                max_value=0.1,
                                value=0.02,
                                step=0.005,
                                format="%.3f",
                                key=f"{key_prefix}_lumo_iso",
                                help="调大：轨道包络面收缩；调小：轨道包络面扩展"
                            )
                        with col2:
                            st.metric("当前值", f"{lumo_iso:.3f}")
                        
                        if len(lumo_files) == 1:
                            # 单个文件
                            lumo_html = create_orbital_viewer(
                                lumo_files[0]['path'],
                                width=650,
                                height=500,
                                iso_value=lumo_iso,
                                orbital_type="LUMO"
                            )
                            components.html(lumo_html, height=560, scrolling=False)
                            st.caption(f"文件: {os.path.basename(lumo_files[0]['path'])}")
                        else:
                            # 多个文件，使用 tabs 切换
                            st.markdown(f"**共 {len(lumo_files)} 个 LUMO 轨道文件**")
                            lumo_tab_names = [f"#{i+1}: {get_structure_label(f['id'], i)}" for i, f in enumerate(lumo_files)]
                            lumo_sub_tabs = st.tabs(lumo_tab_names)
                            
                            for i, (sub_tab, lumo_file) in enumerate(zip(lumo_sub_tabs, lumo_files)):
                                with sub_tab:
                                    lumo_html = create_orbital_viewer(
                                        lumo_file['path'],
                                        width=650,
                                        height=450,
                                        iso_value=lumo_iso,
                                        orbital_type="LUMO"
                                    )
                                    components.html(lumo_html, height=510, scrolling=False)
                                    st.caption(f"文件: {os.path.basename(lumo_file['path'])}")
                    except Exception as e:
                        st.error(f"LUMO 可视化失败: {e}")
            
            # ESP (静电势) Tab
            if tab_esp is not None and has_esp:
                with tab_esp:
                    try:
                        st.markdown("**静电势 (ESP) 可视化**")
                        st.caption("展示分子表面静电势分布：红色为正（亲核区域），蓝色为负（亲电区域）")
                        
                        # 色阶范围控制 (eV)
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            esp_range_ev = st.slider(
                                "色阶范围 (eV)",
                                min_value=0.2,
                                max_value=3.0,
                                value=0.82,  # 默认 0.03 a.u. ≈ 0.82 eV
                                step=0.1,
                                format="%.2f",
                                key=f"{key_prefix}_esp_range",
                                help="调整 ESP 色阶的显示范围，超出范围的值会被截断到边界颜色"
                            )
                        with col2:
                            st.metric("±范围", f"{esp_range_ev:.2f} eV")
                        
                        # 转换为原子单位 (a.u.)
                        HARTREE_TO_EV = 27.2114
                        esp_colorscale_max = esp_range_ev / HARTREE_TO_EV
                        
                        if len(esp_files_list) == 1:
                            # 单个 ESP 文件组
                            esp_files = esp_files_list[0]
                            esp_html = create_esp_viewer(
                                esp_files['density'],
                                esp_files['esp'],
                                esp_files.get('info'),
                                width=650,
                                height=500,
                                esp_colorscale_min=-esp_colorscale_max,
                                esp_colorscale_max=esp_colorscale_max,
                            )
                            components.html(esp_html, height=600, scrolling=False)
                            
                            # 显示文件信息
                            st.caption(f"密度文件: {os.path.basename(esp_files['density'])}")
                            st.caption(f"ESP文件: {os.path.basename(esp_files['esp'])}")
                            
                            # 如果有 ESP info，显示极值信息
                            if esp_files.get('info') and os.path.exists(esp_files['info']):
                                _show_esp_info(esp_files['info'])
                        else:
                            # 多个 ESP 文件组，使用 tabs 切换
                            st.markdown(f"**共 {len(esp_files_list)} 个 ESP 文件**")
                            esp_tab_names = [f"#{i+1}: {get_structure_label(f['id'], i)}" for i, f in enumerate(esp_files_list)]
                            esp_sub_tabs = st.tabs(esp_tab_names)
                            
                            for i, (sub_tab, esp_files) in enumerate(zip(esp_sub_tabs, esp_files_list)):
                                with sub_tab:
                                    esp_html = create_esp_viewer(
                                        esp_files['density'],
                                        esp_files['esp'],
                                        esp_files.get('info'),
                                        width=650,
                                        height=450,
                                        esp_colorscale_min=-esp_colorscale_max,
                                        esp_colorscale_max=esp_colorscale_max,
                                    )
                                    components.html(esp_html, height=510, scrolling=False)
                                    
                                    # 显示文件信息
                                    st.caption(f"密度文件: {os.path.basename(esp_files['density'])}")
                                    
                                    # 如果有 ESP info，显示极值信息
                                    if esp_files.get('info') and os.path.exists(esp_files['info']):
                                        _show_esp_info(esp_files['info'])
                                
                    except Exception as e:
                        st.error(f"ESP 可视化失败: {e}")
            
            # Li Deformation Tab
            if tab_li_deform is not None and li_deform_files:
                with tab_li_deform:
                    try:
                        st.markdown("**Li 离子变形因子可视化**")
                        st.caption("展示 Li 离子周围电子密度变形的等值面分布")
                        
                        # 查找对应的分子结构 xyz 文件
                        # 优先从 task 目录的 xyz_all 中查找
                        task_dir = os.path.dirname(os.path.dirname(infer_dir))
                        xyz_all_dir = os.path.join(task_dir, "xyz_all")
                        
                        # 收集所有 xyz 文件，按编号排序
                        xyz_files_map = {}
                        if os.path.exists(xyz_all_dir):
                            import glob as glob_module
                            xyz_files = glob_module.glob(os.path.join(xyz_all_dir, "*.xyz"))
                            for xf in xyz_files:
                                basename = os.path.basename(xf)
                                # 尝试从文件名提取编号
                                m = re.search(r'(\d+)', basename)
                                if m:
                                    xyz_files_map[m.group(1)] = xf
                                else:
                                    xyz_files_map['0'] = xf
                        
                        # 透明度控制
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            opacity = st.slider(
                                "表面透明度",
                                min_value=0.1,
                                max_value=1.0,
                                value=0.65,
                                step=0.05,
                                format="%.2f",
                                key=f"{key_prefix}_li_deform_opacity",
                                help="调整 Li deformation 表面的透明度"
                            )
                        with col2:
                            st.metric("透明度", f"{opacity:.2f}")
                        
                        if len(li_deform_files) == 1:
                            # 单个 Li deformation 文件
                            li_file = li_deform_files[0]
                            molecule_path = xyz_files_map.get(li_file['id'], list(xyz_files_map.values())[0] if xyz_files_map else None)
                            
                            if molecule_path is None and os.path.exists(db_path):
                                # 从 db 导出
                                from emolagent.visualization import atoms_to_xyz_string
                                atoms = load_structure_from_db(db_path)
                                if atoms:
                                    temp_xyz_path = os.path.join(infer_dir, "temp_molecule.xyz")
                                    with open(temp_xyz_path, 'w') as f:
                                        f.write(atoms_to_xyz_string(atoms, "Generated for Li Deformation visualization"))
                                    molecule_path = temp_xyz_path
                            
                            if molecule_path is None:
                                st.warning("未找到分子结构文件，无法叠加显示")
                            else:
                                li_deform_html = create_li_deformation_viewer(
                                    molecule_path=molecule_path,
                                    surface_pdb_path=li_file['path'],
                                    width=650,
                                    height=500,
                                    surface_opacity=opacity,
                                    isovalue=li_file.get('isovalue', '0.09'),
                                )
                                components.html(li_deform_html, height=560, scrolling=False)
                                st.caption(f"文件: {os.path.basename(li_file['path'])} | 等值面: {li_file.get('isovalue', 'N/A')}")
                        else:
                            # 多个 Li deformation 文件，使用 tabs 切换
                            st.markdown(f"**共 {len(li_deform_files)} 个 Li Deformation 文件**")
                            li_tab_names = [f"#{i+1}: {get_structure_label(f['id'], i)}" for i, f in enumerate(li_deform_files)]
                            li_sub_tabs = st.tabs(li_tab_names)
                            
                            for i, (sub_tab, li_file) in enumerate(zip(li_sub_tabs, li_deform_files)):
                                with sub_tab:
                                    # 根据 li_file 的 id 查找对应的 xyz 文件
                                    molecule_path = xyz_files_map.get(li_file['id'])
                                    if molecule_path is None and xyz_files_map:
                                        molecule_path = list(xyz_files_map.values())[0]
                                    
                                    if molecule_path is None and os.path.exists(db_path):
                                        # 从 db 导出
                                        from emolagent.visualization import atoms_to_xyz_string
                                        atoms_list = load_all_structures_from_db(db_path, max_count=len(li_deform_files))
                                        if atoms_list and i < len(atoms_list):
                                            atoms, _ = atoms_list[i]
                                            temp_xyz_path = os.path.join(infer_dir, f"temp_molecule_{i}.xyz")
                                            with open(temp_xyz_path, 'w') as f:
                                                f.write(atoms_to_xyz_string(atoms, f"Structure {i} for Li Deformation"))
                                            molecule_path = temp_xyz_path
                                    
                                    if molecule_path is None:
                                        st.warning("未找到分子结构文件，无法叠加显示")
                                    else:
                                        li_deform_html = create_li_deformation_viewer(
                                            molecule_path=molecule_path,
                                            surface_pdb_path=li_file['path'],
                                            width=650,
                                            height=450,
                                            surface_opacity=opacity,
                                            isovalue=li_file.get('isovalue', '0.09'),
                                        )
                                        components.html(li_deform_html, height=510, scrolling=False)
                                        st.caption(f"文件: {os.path.basename(li_file['path'])} | 等值面: {li_file.get('isovalue', 'N/A')}")
                    except Exception as e:
                        st.error(f"Li Deformation 可视化失败: {e}")

        elif structure_match:
            db_path = structure_match.group(1).strip()
            if os.path.exists(db_path):
                # 获取数据库中的结构总数
                total_count = get_structure_count_from_db(db_path)
                max_display = 3  # 最多显示 3 个结构
                
                if total_count <= 1:
                    # 单个结构：保持原有逻辑
                    st.markdown("### 📊 结构预览")
                    
                    with st.expander("🔬 点击查看 3D 分子结构 (可交互)", expanded=True):
                        try:
                            atoms = load_structure_from_db(db_path)
                            if atoms:
                                viewer_html = create_gaussian_view_style_viewer(
                                    atoms,
                                    width=650,
                                    height=500,
                                    style="sphere+stick",
                                    add_lighting=True
                                )
                                components.html(viewer_html, height=550, scrolling=False)
                                
                                st.caption(f"📁 结构路径: `{db_path}`")
                                st.info("💡 提示：您可以说「对上面生成的结构进行电子结构分析」来继续分析")
                            else:
                                st.warning("无法加载结构预览")
                        except Exception as e:
                            st.error(f"结构预览失败: {e}")
                else:
                    # 多个结构：使用 tabs 展示
                    displayed_count = min(total_count, max_display)
                    st.markdown(f"### 📊 结构预览 (共 {total_count} 个团簇，显示前 {displayed_count} 个)")
                    
                    try:
                        structures = load_all_structures_from_db(db_path, max_count=max_display)
                        
                        if structures:
                            # 生成 tab 名称
                            tab_names = []
                            for i, (atoms, meta) in enumerate(structures):
                                solv_name = meta.get('solvent_name', '')
                                anion_name = meta.get('anion_name', '')
                                n_solv = meta.get('n_solv', 0)
                                n_anion = meta.get('n_anion', 0)
                                category = meta.get('category', '')
                                
                                # 构建简洁的标签
                                if anion_name and n_anion > 0:
                                    label = f"{n_solv}{solv_name}+{n_anion}{anion_name}"
                                else:
                                    label = f"{n_solv}{solv_name}"
                                if category:
                                    label = f"[{category}] {label}"
                                tab_names.append(f"结构{i+1}: {label}")
                            
                            tabs = st.tabs(tab_names)
                            
                            for i, (tab, (atoms, meta)) in enumerate(zip(tabs, structures)):
                                with tab:
                                    viewer_html = create_gaussian_view_style_viewer(
                                        atoms,
                                        width=650,
                                        height=500,
                                        style="sphere+stick",
                                        add_lighting=True
                                    )
                                    components.html(viewer_html, height=550, scrolling=False)
                                    
                                    # 显示配方信息
                                    solv_name = meta.get('solvent_name', 'Unknown')
                                    anion_name = meta.get('anion_name', '')
                                    n_solv = meta.get('n_solv', 0)
                                    n_anion = meta.get('n_anion', 0)
                                    ion = meta.get('ion', 'Li')
                                    
                                    formula_parts = [f"1x{ion}⁺"]
                                    if n_solv > 0:
                                        formula_parts.append(f"{n_solv}x{solv_name}")
                                    if n_anion > 0 and anion_name:
                                        formula_parts.append(f"{n_anion}x{anion_name}⁻")
                                    
                                    st.caption(f"配方: {' + '.join(formula_parts)} | 化学式: {atoms.get_chemical_formula()} | 原子数: {len(atoms)}")
                            
                            # 如果有更多未显示的结构
                            if total_count > max_display:
                                st.info(f"💡 还有 {total_count - max_display} 个团簇未显示。完整结果请下载分析包查看。")
                            
                            st.caption(f"📁 结构路径: `{db_path}`")
                            st.info("💡 提示：您可以说「对上面生成的结构进行电子结构分析」来继续分析")
                        else:
                            st.warning("无法加载结构预览")
                    except Exception as e:
                        st.error(f"结构预览失败: {e}")

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
    """主函数。"""

    schedule_cleanup()
    cookie_manager = get_manager()

    if "user" not in st.session_state:
        st.session_state["user"] = None

    if st.session_state["user"] is None and not st.session_state.get("logout_flag", False):
        token = cookie_manager.get("auth_token")
        if token:
            user_info = db.verify_jwt_token(token)
            if user_info:
                st.session_state["user"] = user_info
                st.session_state["current_chat_id"] = None

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
        if st.button("+ 新建对话", type="primary", use_container_width=True):
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
                    chat_folder = os.path.join(get_project_root(), "users", safe_username, "output", str(chat["id"]))
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
            ["gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-2.5-pro"],
            index=0,
        )
        api_key = os.getenv("GOOGLE_API_KEY", "")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0)

        st.markdown("---")
        st.header("📚 知识库管理")
        
        try:
            kb_stats = get_index_stats(api_key)
            if "error" not in kb_stats:
                st.metric("已索引文档块", kb_stats.get("total_documents", 0))
                st.caption(f"已索引文件数: {kb_stats.get('indexed_files', 0)}")
            else:
                st.warning("知识库未初始化")
        except Exception:
            st.warning("知识库未初始化")
        
        if current_user.get("username") in ADMIN_USERS:
            col_idx1, col_idx2 = st.columns(2)
            with col_idx1:
                if st.button("🔄 增量更新", use_container_width=True):
                    with st.spinner("正在更新知识库索引..."):
                        try:
                            stats = build_index(api_key, force_rebuild=False)
                            st.success(
                                f"索引完成！\n"
                                f"新增: {stats['new_indexed']}, "
                                f"跳过: {stats['skipped']}, "
                                f"失败: {stats['failed']}"
                            )
                        except Exception as e:
                            st.error(f"索引失败: {e}")
            
            with col_idx2:
                if st.button("🔨 重建索引", use_container_width=True):
                    with st.spinner("正在重建知识库索引（这可能需要几分钟）..."):
                        try:
                            stats = build_index(api_key, force_rebuild=True)
                            st.success(
                                f"重建完成！\n"
                                f"共索引 {stats['new_indexed']} 个文件, "
                                f"{stats['total_chunks']} 个文档块"
                            )
                        except Exception as e:
                            st.error(f"索引失败: {e}")

        st.markdown("---")
        st.header("🖥️ GPU 任务状态")
        try:
            queue_status = get_task_queue_status()
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                st.metric("运行中", f"{queue_status['active_tasks']}")
            with col_q2:
                st.metric("最大并发", f"{queue_status['max_tasks']}")
            
            # 显示每张 GPU 的负载
            gpu_loads = queue_status.get('gpu_loads', {})
            if gpu_loads:
                st.caption("**GPU 负载分布:**")
                gpu_cols = st.columns(len(gpu_loads))
                for i, (gpu_id, load) in enumerate(sorted(gpu_loads.items())):
                    with gpu_cols[i]:
                        max_per_gpu = queue_status['max_tasks'] // len(gpu_loads)
                        st.metric(f"GPU {gpu_id}", f"{load}/{max_per_gpu}")
            
            if queue_status['can_accept']:
                st.success(f"✅ 可接受新任务 (剩余 {queue_status['available_slots']} 槽位)")
            else:
                st.warning(f"⏳ 队列已满，新任务需排队等待")
        except Exception as e:
            st.caption(f"无法获取队列状态: {e}")

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
        st.info("暂无对话，请在左侧点击 [+ 新建对话] 按钮。")
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

                output_text = None
                msgs = response.get("messages") if isinstance(response, dict) else None
                if msgs and isinstance(msgs, list):
                    last = msgs[-1]
                    if isinstance(last, dict):
                        output_text = last.get("content")
                    else:
                        output_text = getattr(last, "content", None)

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


if __name__ == "__main__":
    main()
