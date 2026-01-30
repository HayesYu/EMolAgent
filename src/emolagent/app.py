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
from emolagent.utils.config import ModelConfig, AuthConfig, KnowledgeConfig
from emolagent.utils.i18n import t, get_welcome_message, get_system_prompt

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


# 配置参数（从配置文件加载）
DEFAULT_MODEL_PATH = ModelConfig.get_inference_model_path()
ADMIN_USERS = AuthConfig.get_admin_users()
LITERATURE_PATH = KnowledgeConfig.get_literature_path()

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
    # 如果 model_path 为空、为 "None"、或路径不存在，使用默认模型路径
    if model_path in ["None", "", None] or not os.path.exists(model_path):
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


def build_agent(model_name: str, temperature: float, api_key: str, lang: str = "zh"):
    """构建并返回 LangChain agent。"""
    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        timeout=30,
        max_output_tokens=2000,
    )

    checkpointer = get_checkpointer()
    
    # 根据语言获取系统提示词
    system_prompt = get_system_prompt(lang)

    agent = create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=system_prompt,
        context_schema=Context,
        checkpointer=checkpointer,
    )
    return agent


# ==============================================================================
# 4. UI：登录 / 主界面
# ==============================================================================

def login_ui(cookie_manager):
    """处理登录和注册的 UI 渲染。"""
    # 初始化语言状态（从 Cookie 读取）
    if "language" not in st.session_state:
        saved_lang = cookie_manager.get("user_language")
        st.session_state["language"] = saved_lang if saved_lang in ["zh", "en"] else "zh"
    
    lang = st.session_state["language"]
    
    # 标题行：左侧标题，右侧语言切换
    col_title, col_lang = st.columns([5, 1])
    with col_title:
        st.title(f"🧪 {t('app_title', lang)} - {t('please_login', lang)}")
    with col_lang:
        lang_options = ["中文", "English"]
        current_idx = 0 if lang == "zh" else 1
        selected_lang = st.selectbox(
            "🌐",
            lang_options,
            index=current_idx,
            key="login_lang_selector",
            label_visibility="collapsed",
        )
        new_lang = "zh" if selected_lang == "中文" else "en"
        if new_lang != lang:
            st.session_state["language"] = new_lang
            expires = datetime.datetime.now() + datetime.timedelta(days=30)
            cookie_manager.set("user_language", new_lang, expires_at=expires)
            st.rerun()
    
    tab1, tab2 = st.tabs([t("login", lang), t("register", lang)])

    with tab1:
        with st.form("login_form"):
            username = st.text_input(t("username", lang))
            password = st.text_input(t("password", lang), type="password")
            submitted = st.form_submit_button(t("login", lang))
            if submitted:
                user = db.login_user(username, password)
                if user:
                    st.session_state["user"] = user
                    st.session_state["current_chat_id"] = None
                    st.session_state["logout_flag"] = False
                    token = db.create_jwt_token(user["id"], user["username"])
                    expires = datetime.datetime.now() + datetime.timedelta(days=3)
                    cookie_manager.set("auth_token", token, expires_at=expires)
                    st.success(t("login_success", lang))
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(t("login_failed", lang))

    with tab2:
        with st.form("register_form"):
            new_user = st.text_input(t("new_username", lang))
            new_pass = st.text_input(t("new_password", lang), type="password")
            confirm_pass = st.text_input(t("confirm_password", lang), type="password")
            submitted = st.form_submit_button(t("register", lang))
            if submitted:
                if new_user and new_pass and confirm_pass:
                    if new_pass != confirm_pass:
                        st.error(t("password_mismatch", lang))
                    elif db.register_user(new_user, new_pass):
                        st.success(t("register_success", lang))
                    else:
                        st.error(t("username_exists", lang))
                else:
                    st.error(t("enter_username_password", lang))


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


def _show_esp_info(info_path: str, lang: str = "zh"):
    """显示 ESP 极值信息的辅助函数。"""
    try:
        import json as json_module
        with open(info_path, 'r') as f:
            esp_info = json_module.load(f)
        
        st.markdown("---")
        st.markdown(f"**{t('esp_extrema_info', lang)}**")
        col_max, col_min = st.columns(2)
        with col_max:
            max_val = esp_info.get('ESP_max_eV', 'N/A')
            max_loc = esp_info.get('ESP_max_location_Ang', [])
            st.metric(t("max_value", lang), f"{max_val:.4f}" if isinstance(max_val, (int, float)) else max_val)
            if max_loc:
                st.caption(f"{t('location', lang)}: ({max_loc[0]:.2f}, {max_loc[1]:.2f}, {max_loc[2]:.2f}) Å")
        with col_min:
            min_val = esp_info.get('ESP_min_eV', 'N/A')
            min_loc = esp_info.get('ESP_min_location_Ang', [])
            st.metric(t("min_value", lang), f"{min_val:.4f}" if isinstance(min_val, (int, float)) else min_val)
            if min_loc:
                st.caption(f"{t('location', lang)}: ({min_loc[0]:.2f}, {min_loc[1]:.2f}, {min_loc[2]:.2f}) Å")
    except Exception:
        pass


def render_message_with_download(role: str, content: Any, key_prefix: str, lang: str = "zh"):
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
            
            st.markdown(f"### 🔬 {t('analysis_results', lang)}")
            
            # 查找 Li deformation 文件和 ESP 文件
            li_deform_files = find_li_deformation_files(infer_dir)
            esp_files_list = find_esp_files(infer_dir)
            has_esp = len(esp_files_list) > 0
            
            # 查找轨道文件
            orbital_files = find_orbital_files(infer_dir)
            has_homo = len(orbital_files.get('homo', [])) > 0
            has_lumo = len(orbital_files.get('lumo', [])) > 0
            
            # 根据可用文件决定 tab 数量
            tab_names = [f"🧬 {t('cluster_structure', lang)}"]
            if has_homo:
                tab_names.append(f"🔵 {t('homo_orbital', lang)}")
            if has_lumo:
                tab_names.append(f"🟢 {t('lumo_orbital', lang)}")
            if has_esp:
                tab_names.append(f"⚡ {t('esp', lang)}")
            if li_deform_files:
                tab_names.append(f"💠 {t('li_deformation', lang)}")
            
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
                struct_prefix = "Structure" if lang == "en" else "结构"
                return f"{struct_prefix}{file_id}"
            
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
                                    add_lighting=True,
                                    lang=lang
                                )
                                components.html(viewer_html, height=560, scrolling=False)
                                st.caption(f"{t('chemical_formula', lang)}: {atoms.get_chemical_formula()} | {t('atom_count', lang)}: {len(atoms)}")
                            else:
                                st.warning(t("cannot_load_preview", lang))
                        else:
                            # 多个结构
                            displayed_count = min(total_count, max_display)
                            st.markdown(f"**{t('total_clusters', lang, total=total_count, displayed=displayed_count)}**")
                            
                            structures_to_show = structures_data[:max_display]
                            
                            if structures_to_show:
                                # 生成子 tab 名称，复用 structure_labels
                                struct_prefix = "Structure" if lang == "en" else "结构"
                                sub_tab_names = [f"#{i+1}: {structure_labels.get(str(i), f'{struct_prefix}{i}')}" 
                                                 for i in range(len(structures_to_show))]
                                
                                sub_tabs = st.tabs(sub_tab_names)
                                
                                for i, (sub_tab, (atoms, meta)) in enumerate(zip(sub_tabs, structures_to_show)):
                                    with sub_tab:
                                        viewer_html = create_gaussian_view_style_viewer(
                                            atoms,
                                            width=650,
                                            height=450,
                                            style="sphere+stick",
                                            add_lighting=True,
                                            lang=lang
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
                                        
                                        st.caption(f"{t('formula', lang)}: {' + '.join(formula_parts)} | {t('chemical_formula', lang)}: {atoms.get_chemical_formula()} | {t('atom_count', lang)}: {len(atoms)}")
                                
                                if total_count > max_display:
                                    st.info(f"💡 {t('more_structures_hint', lang, count=total_count - max_display)}")
                            else:
                                st.warning(t("cannot_load_preview", lang))
                    except Exception as e:
                        st.error(t("preview_failed", lang, error=str(e)))
                else:
                    st.warning(t("file_not_exist", lang, path=db_path))
            
            # HOMO Tab
            if tab_homo is not None and has_homo:
                with tab_homo:
                    homo_files = orbital_files.get('homo', [])
                    try:
                        # 等值面设置
                        st.markdown(f"**{t('isovalue_settings', lang)}**")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            homo_iso = st.slider(
                                t("isovalue_size", lang),
                                min_value=0.005,
                                max_value=0.1,
                                value=0.02,
                                step=0.005,
                                format="%.3f",
                                key=f"{key_prefix}_homo_iso",
                                help=t("isovalue_hint", lang)
                            )
                        with col2:
                            st.metric(t("current_value", lang), f"{homo_iso:.3f}")
                        
                        if len(homo_files) == 1:
                            # 单个文件
                            homo_html = create_orbital_viewer(
                                homo_files[0]['path'],
                                width=650,
                                height=500,
                                iso_value=homo_iso,
                                orbital_type="HOMO",
                                lang=lang
                            )
                            components.html(homo_html, height=560, scrolling=False)
                            st.caption(f"{t('file', lang)}: {os.path.basename(homo_files[0]['path'])}")
                        else:
                            # 多个文件，使用 tabs 切换
                            st.markdown(f"**{len(homo_files)} HOMO {'files' if lang == 'en' else '轨道文件'}**")
                            homo_tab_names = [f"#{i+1}: {get_structure_label(f['id'], i)}" for i, f in enumerate(homo_files)]
                            homo_sub_tabs = st.tabs(homo_tab_names)
                            
                            for i, (sub_tab, homo_file) in enumerate(zip(homo_sub_tabs, homo_files)):
                                with sub_tab:
                                    homo_html = create_orbital_viewer(
                                        homo_file['path'],
                                        width=650,
                                        height=450,
                                        iso_value=homo_iso,
                                        orbital_type="HOMO",
                                        lang=lang
                                    )
                                    components.html(homo_html, height=510, scrolling=False)
                                    st.caption(f"{t('file', lang)}: {os.path.basename(homo_file['path'])}")
                    except Exception as e:
                        st.error(t("homo_vis_failed", lang, error=str(e)))
            
            # LUMO Tab
            if tab_lumo is not None and has_lumo:
                with tab_lumo:
                    lumo_files = orbital_files.get('lumo', [])
                    try:
                        # 等值面设置
                        st.markdown(f"**{t('isovalue_settings', lang)}**")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            lumo_iso = st.slider(
                                t("isovalue_size", lang),
                                min_value=0.005,
                                max_value=0.1,
                                value=0.02,
                                step=0.005,
                                format="%.3f",
                                key=f"{key_prefix}_lumo_iso",
                                help=t("isovalue_hint", lang)
                            )
                        with col2:
                            st.metric(t("current_value", lang), f"{lumo_iso:.3f}")
                        
                        if len(lumo_files) == 1:
                            # 单个文件
                            lumo_html = create_orbital_viewer(
                                lumo_files[0]['path'],
                                width=650,
                                height=500,
                                iso_value=lumo_iso,
                                orbital_type="LUMO",
                                lang=lang
                            )
                            components.html(lumo_html, height=560, scrolling=False)
                            st.caption(f"{t('file', lang)}: {os.path.basename(lumo_files[0]['path'])}")
                        else:
                            # 多个文件，使用 tabs 切换
                            st.markdown(f"**{len(lumo_files)} LUMO {'files' if lang == 'en' else '轨道文件'}**")
                            lumo_tab_names = [f"#{i+1}: {get_structure_label(f['id'], i)}" for i, f in enumerate(lumo_files)]
                            lumo_sub_tabs = st.tabs(lumo_tab_names)
                            
                            for i, (sub_tab, lumo_file) in enumerate(zip(lumo_sub_tabs, lumo_files)):
                                with sub_tab:
                                    lumo_html = create_orbital_viewer(
                                        lumo_file['path'],
                                        width=650,
                                        height=450,
                                        iso_value=lumo_iso,
                                        orbital_type="LUMO",
                                        lang=lang
                                    )
                                    components.html(lumo_html, height=510, scrolling=False)
                                    st.caption(f"{t('file', lang)}: {os.path.basename(lumo_file['path'])}")
                    except Exception as e:
                        st.error(t("lumo_vis_failed", lang, error=str(e)))
            
            # ESP (静电势) Tab
            if tab_esp is not None and has_esp:
                with tab_esp:
                    try:
                        st.markdown(f"**{t('esp_visualization', lang)}**")
                        st.caption(t("esp_description", lang))
                        
                        # 色阶范围控制 (eV)
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            esp_range_ev = st.slider(
                                t("color_scale_range", lang),
                                min_value=0.2,
                                max_value=3.0,
                                value=0.82,  # 默认 0.03 a.u. ≈ 0.82 eV
                                step=0.1,
                                format="%.2f",
                                key=f"{key_prefix}_esp_range",
                                help=t("color_scale_hint", lang)
                            )
                        with col2:
                            st.metric(t("range", lang), f"{esp_range_ev:.2f} eV")
                        
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
                                lang=lang,
                            )
                            components.html(esp_html, height=600, scrolling=False)
                            
                            # 显示文件信息
                            st.caption(f"{t('density_file', lang)}: {os.path.basename(esp_files['density'])}")
                            st.caption(f"{t('esp_file', lang)}: {os.path.basename(esp_files['esp'])}")
                            
                            # 如果有 ESP info，显示极值信息
                            if esp_files.get('info') and os.path.exists(esp_files['info']):
                                _show_esp_info(esp_files['info'], lang)
                        else:
                            # 多个 ESP 文件组，使用 tabs 切换
                            st.markdown(f"**{len(esp_files_list)} ESP {'files' if lang == 'en' else '文件'}**")
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
                                        lang=lang,
                                    )
                                    components.html(esp_html, height=510, scrolling=False)
                                    
                                    # 显示文件信息
                                    st.caption(f"{t('density_file', lang)}: {os.path.basename(esp_files['density'])}")
                                    
                                    # 如果有 ESP info，显示极值信息
                                    if esp_files.get('info') and os.path.exists(esp_files['info']):
                                        _show_esp_info(esp_files['info'], lang)
                                
                    except Exception as e:
                        st.error(t("esp_vis_failed", lang, error=str(e)))
            
            # Li Deformation Tab
            if tab_li_deform is not None and li_deform_files:
                with tab_li_deform:
                    try:
                        st.markdown(f"**{t('li_deformation_visualization', lang)}**")
                        st.caption(t("li_deformation_description", lang))
                        
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
                                t("surface_opacity", lang),
                                min_value=0.1,
                                max_value=1.0,
                                value=0.65,
                                step=0.05,
                                format="%.2f",
                                key=f"{key_prefix}_li_deform_opacity",
                                help=t("opacity_hint", lang)
                            )
                        with col2:
                            st.metric(t("opacity", lang), f"{opacity:.2f}")
                        
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
                                st.warning(t("no_molecule_file", lang))
                            else:
                                li_deform_html = create_li_deformation_viewer(
                                    molecule_path=molecule_path,
                                    surface_pdb_path=li_file['path'],
                                    width=650,
                                    height=500,
                                    surface_opacity=opacity,
                                    isovalue=li_file.get('isovalue', '0.09'),
                                    lang=lang,
                                )
                                components.html(li_deform_html, height=560, scrolling=False)
                                st.caption(f"{t('file', lang)}: {os.path.basename(li_file['path'])} | {t('isovalue', lang)}: {li_file.get('isovalue', 'N/A')}")
                        else:
                            # 多个 Li deformation 文件，使用 tabs 切换
                            st.markdown(f"**{len(li_deform_files)} Li Deformation {'files' if lang == 'en' else '文件'}**")
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
                                        st.warning(t("no_molecule_file", lang))
                                    else:
                                        li_deform_html = create_li_deformation_viewer(
                                            molecule_path=molecule_path,
                                            surface_pdb_path=li_file['path'],
                                            width=650,
                                            height=450,
                                            surface_opacity=opacity,
                                            isovalue=li_file.get('isovalue', '0.09'),
                                            lang=lang,
                                        )
                                        components.html(li_deform_html, height=510, scrolling=False)
                                        st.caption(f"{t('file', lang)}: {os.path.basename(li_file['path'])} | {t('isovalue', lang)}: {li_file.get('isovalue', 'N/A')}")
                    except Exception as e:
                        st.error(t("li_deform_vis_failed", lang, error=str(e)))

        elif structure_match:
            db_path = structure_match.group(1).strip()
            if os.path.exists(db_path):
                # 获取数据库中的结构总数
                total_count = get_structure_count_from_db(db_path)
                max_display = 3  # 最多显示 3 个结构
                
                if total_count <= 1:
                    # 单个结构：保持原有逻辑
                    st.markdown(f"### 📊 {t('structure_preview', lang)}")
                    
                    with st.expander(f"🔬 {t('view_3d_structure', lang)}", expanded=True):
                        try:
                            atoms = load_structure_from_db(db_path)
                            if atoms:
                                viewer_html = create_gaussian_view_style_viewer(
                                    atoms,
                                    width=650,
                                    height=500,
                                    style="sphere+stick",
                                    add_lighting=True,
                                    lang=lang
                                )
                                components.html(viewer_html, height=550, scrolling=False)
                                
                                st.caption(f"📁 {t('structure_path', lang)}: `{db_path}`")
                                st.info(f"💡 {t('continue_analysis_hint', lang)}")
                            else:
                                st.warning(t("cannot_load_preview", lang))
                        except Exception as e:
                            st.error(t("preview_failed", lang, error=str(e)))
                else:
                    # 多个结构：使用 tabs 展示
                    displayed_count = min(total_count, max_display)
                    st.markdown(f"### 📊 {t('structure_preview', lang)} ({t('total_clusters', lang, total=total_count, displayed=displayed_count)})")
                    
                    try:
                        structures = load_all_structures_from_db(db_path, max_count=max_display)
                        
                        if structures:
                            # 生成 tab 名称
                            tab_names = []
                            struct_prefix = "Structure" if lang == "en" else "结构"
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
                                tab_names.append(f"{struct_prefix}{i+1}: {label}")
                            
                            tabs = st.tabs(tab_names)
                            
                            for i, (tab, (atoms, meta)) in enumerate(zip(tabs, structures)):
                                with tab:
                                    viewer_html = create_gaussian_view_style_viewer(
                                        atoms,
                                        width=650,
                                        height=500,
                                        style="sphere+stick",
                                        add_lighting=True,
                                        lang=lang
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
                                    
                                    st.caption(f"{t('formula', lang)}: {' + '.join(formula_parts)} | {t('chemical_formula', lang)}: {atoms.get_chemical_formula()} | {t('atom_count', lang)}: {len(atoms)}")
                            
                            # 如果有更多未显示的结构
                            if total_count > max_display:
                                st.info(f"💡 {t('more_structures_hint', lang, count=total_count - max_display)}")
                            
                            st.caption(f"📁 {t('structure_path', lang)}: `{db_path}`")
                            st.info(f"💡 {t('continue_analysis_hint', lang)}")
                        else:
                            st.warning(t("cannot_load_preview", lang))
                    except Exception as e:
                        st.error(t("preview_failed", lang, error=str(e)))

        if download_match:
            file_path = download_match.group(1).strip()
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    st.download_button(
                        label=f"📦 {t('download_results', lang)}",
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
    
    # 初始化语言状态（从 Cookie 读取，如果 login_ui 已设置则保持）
    if "language" not in st.session_state:
        saved_lang = cookie_manager.get("user_language")
        st.session_state["language"] = saved_lang if saved_lang in ["zh", "en"] else "zh"
    
    lang = st.session_state["language"]

    # 1. Sidebar
    with st.sidebar:
        # 用户信息和语言切换
        col_user, col_lang = st.columns([3, 2])
        with col_user:
            st.write(f"👤 **{current_user['username']}**")
        with col_lang:
            lang_options = ["中文", "English"]
            current_idx = 0 if lang == "zh" else 1
            selected_lang = st.selectbox(
                "🌐",
                lang_options,
                index=current_idx,
                key="main_lang_selector",
                label_visibility="collapsed",
            )
            new_lang = "zh" if selected_lang == "中文" else "en"
            if new_lang != lang:
                st.session_state["language"] = new_lang
                lang = new_lang
                expires = datetime.datetime.now() + datetime.timedelta(days=30)
                cookie_manager.set("user_language", new_lang, expires_at=expires)
                st.rerun()
        
        if st.button(t("logout", lang), type="secondary"):
            st.session_state["user"] = None
            st.session_state["messages"] = []
            st.session_state["current_chat_id"] = None
            st.session_state["logout_flag"] = True
            cookie_manager.delete("auth_token")
            st.rerun()

        st.markdown("---")
        if st.button(t("new_chat", lang), type="primary", use_container_width=True):
            st.session_state["suppress_autocreate"] = False
            new_id = db.create_conversation(current_user["id"], title="New Chat")
            st.session_state["current_chat_id"] = new_id
            welcome_msg = get_welcome_message(lang)
            st.session_state["messages"] = [{"role": "assistant", "content": welcome_msg}]
            db.add_message(new_id, "assistant", welcome_msg)
            st.rerun()

        st.markdown(f"### 🕒 {t('history', lang)}")
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
                            st.toast(f"⚠️ {t('folder_delete_failed', lang, error=str(e))}")
                    db.delete_conversation(chat["id"])
                    if st.session_state.get("current_chat_id") == chat["id"]:
                        st.session_state["current_chat_id"] = None
                        st.session_state["messages"] = []
                    st.rerun()

        st.markdown("---")
        st.header(t("model_settings", lang))
        model_name = st.selectbox(
            t("select_model", lang),
            ["gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-2.5-pro"],
            index=0,
        )
        api_key = os.getenv("GOOGLE_API_KEY", "")
        temperature = st.slider(t("temperature", lang), 0.0, 1.0, 0.0)

        st.markdown("---")
        st.header(f"📚 {t('knowledge_base', lang)}")
        
        try:
            kb_stats = get_index_stats(api_key)
            if "error" not in kb_stats:
                st.metric(t("indexed_docs", lang), kb_stats.get("total_documents", 0))
                st.caption(f"{t('indexed_files', lang)}: {kb_stats.get('indexed_files', 0)}")
            else:
                st.warning(t("kb_not_initialized", lang))
        except Exception:
            st.warning(t("kb_not_initialized", lang))
        
        if current_user.get("username") in ADMIN_USERS:
            col_idx1, col_idx2 = st.columns(2)
            with col_idx1:
                if st.button(f"🔄 {t('incremental_update', lang)}", use_container_width=True):
                    with st.spinner(t("updating_kb", lang)):
                        try:
                            stats = build_index(api_key, force_rebuild=False)
                            st.success(
                                t("index_complete", lang, 
                                  new=stats['new_indexed'], 
                                  skipped=stats['skipped'], 
                                  failed=stats['failed'])
                            )
                        except Exception as e:
                            st.error(t("index_failed", lang, error=str(e)))
            
            with col_idx2:
                if st.button(f"🔨 {t('rebuild_index', lang)}", use_container_width=True):
                    with st.spinner(t("rebuilding_kb", lang)):
                        try:
                            stats = build_index(api_key, force_rebuild=True)
                            st.success(
                                t("rebuild_complete", lang,
                                  files=stats['new_indexed'],
                                  chunks=stats['total_chunks'])
                            )
                        except Exception as e:
                            st.error(t("index_failed", lang, error=str(e)))

        st.markdown("---")
        st.header(f"🖥️ {t('gpu_status', lang)}")
        try:
            queue_status = get_task_queue_status()
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                st.metric(t("running", lang), f"{queue_status['active_tasks']}")
            with col_q2:
                st.metric(t("max_concurrent", lang), f"{queue_status['max_tasks']}")
            
            # 显示每张 GPU 的负载
            gpu_loads = queue_status.get('gpu_loads', {})
            if gpu_loads:
                st.caption(f"**{t('gpu_load', lang)}**")
                gpu_cols = st.columns(len(gpu_loads))
                for i, (gpu_id, load) in enumerate(sorted(gpu_loads.items())):
                    with gpu_cols[i]:
                        max_per_gpu = queue_status['max_tasks'] // len(gpu_loads)
                        st.metric(f"GPU {gpu_id}", f"{load}/{max_per_gpu}")
            
            if queue_status['can_accept']:
                st.success(f"✅ {t('can_accept_task', lang, slots=queue_status['available_slots'])}")
            else:
                st.warning(f"⏳ {t('queue_full', lang)}")
        except Exception as e:
            st.caption(t("cannot_get_queue_status", lang, error=str(e)))

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
                welcome_msg = get_welcome_message(lang)
                st.session_state["messages"] = [{"role": "assistant", "content": welcome_msg}]
                db.add_message(new_id, "assistant", welcome_msg)

    if st.session_state.get("current_chat_id") is None:
        st.title(f"🧪 {t('app_title', lang)}")
        st.info(t("no_chat", lang))
        return

    # 3. LLM Setup
    if not api_key:
        st.warning(f"⚠️ {t('api_key_invalid', lang)}")
        st.stop()

    try:
        agent = build_agent(model_name=model_name, temperature=temperature, api_key=api_key, lang=lang)
    except Exception as e:
        st.error(t("agent_init_failed", lang, error=str(e)))
        st.stop()

    # 4. Chat Interface
    st.title(f"🧪 {t('app_title', lang)}")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for idx, msg in enumerate(st.session_state["messages"]):
        render_message_with_download(
            role=msg["role"],
            content=msg["content"],
            key_prefix=f"history_{idx}",
            lang=lang,
        )

    # 5. Handle Input
    if prompt_input := st.chat_input(t("input_placeholder", lang)):
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

        with st.spinner(t("thinking", lang)):
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
                    lang=lang,
                )

                st.session_state.messages.append({"role": "assistant", "content": output_text_str})
                db.add_message(current_chat_id, "assistant", output_text_str)

            except Exception as e:
                error_msg = t("execution_error", lang, error=str(e))
                render_message_with_download(
                    role="assistant",
                    content=error_msg,
                    key_prefix="current_run_error",
                    lang=lang,
                )
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                db.add_message(current_chat_id, "assistant", error_msg)


if __name__ == "__main__":
    main()
