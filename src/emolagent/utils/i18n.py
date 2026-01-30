"""
EMolAgent 国际化 (i18n) 模块

提供中英文双语支持。
"""

from typing import Literal

Language = Literal["zh", "en"]

# ==============================================================================
# 翻译字典
# ==============================================================================

TRANSLATIONS: dict[str, dict[Language, str]] = {
    # ==========================================
    # 登录/注册页面
    # ==========================================
    "app_title": {
        "zh": "EMolAgent",
        "en": "EMolAgent",
    },
    "please_login": {
        "zh": "请先登录",
        "en": "Please Login",
    },
    "login": {
        "zh": "登录",
        "en": "Login",
    },
    "register": {
        "zh": "注册",
        "en": "Register",
    },
    "username": {
        "zh": "用户名",
        "en": "Username",
    },
    "password": {
        "zh": "密码",
        "en": "Password",
    },
    "new_username": {
        "zh": "新用户名",
        "en": "New Username",
    },
    "new_password": {
        "zh": "新密码",
        "en": "New Password",
    },
    "confirm_password": {
        "zh": "确认密码",
        "en": "Confirm Password",
    },
    "login_success": {
        "zh": "登录成功！",
        "en": "Login successful!",
    },
    "login_failed": {
        "zh": "用户名或密码错误",
        "en": "Invalid username or password",
    },
    "register_success": {
        "zh": "注册成功！请切换到登录标签页进行登录。",
        "en": "Registration successful! Please switch to the login tab.",
    },
    "username_exists": {
        "zh": "用户名已存在",
        "en": "Username already exists",
    },
    "password_mismatch": {
        "zh": "两次输入的密码不一致",
        "en": "Passwords do not match",
    },
    "enter_username_password": {
        "zh": "请输入用户名和密码",
        "en": "Please enter username and password",
    },
    
    # ==========================================
    # 侧边栏
    # ==========================================
    "logout": {
        "zh": "登出",
        "en": "Logout",
    },
    "new_chat": {
        "zh": "+ 新建对话",
        "en": "+ New Chat",
    },
    "history": {
        "zh": "历史记录",
        "en": "History",
    },
    "model_settings": {
        "zh": "模型设置",
        "en": "Model Settings",
    },
    "select_model": {
        "zh": "选择模型",
        "en": "Select Model",
    },
    "temperature": {
        "zh": "Temperature",
        "en": "Temperature",
    },
    "knowledge_base": {
        "zh": "知识库管理",
        "en": "Knowledge Base",
    },
    "indexed_docs": {
        "zh": "已索引文档块",
        "en": "Indexed Documents",
    },
    "indexed_files": {
        "zh": "已索引文件数",
        "en": "Indexed Files",
    },
    "kb_not_initialized": {
        "zh": "知识库未初始化",
        "en": "Knowledge base not initialized",
    },
    "incremental_update": {
        "zh": "增量更新",
        "en": "Update",
    },
    "rebuild_index": {
        "zh": "重建索引",
        "en": "Rebuild",
    },
    "updating_kb": {
        "zh": "正在更新知识库索引...",
        "en": "Updating knowledge base index...",
    },
    "rebuilding_kb": {
        "zh": "正在重建知识库索引（这可能需要几分钟）...",
        "en": "Rebuilding knowledge base index (this may take a few minutes)...",
    },
    "index_complete": {
        "zh": "索引完成！新增: {new}, 跳过: {skipped}, 失败: {failed}",
        "en": "Indexing complete! New: {new}, Skipped: {skipped}, Failed: {failed}",
    },
    "rebuild_complete": {
        "zh": "重建完成！共索引 {files} 个文件, {chunks} 个文档块",
        "en": "Rebuild complete! Indexed {files} files, {chunks} document chunks",
    },
    "index_failed": {
        "zh": "索引失败: {error}",
        "en": "Indexing failed: {error}",
    },
    "gpu_status": {
        "zh": "GPU 任务状态",
        "en": "GPU Task Status",
    },
    "running": {
        "zh": "运行中",
        "en": "Running",
    },
    "max_concurrent": {
        "zh": "最大并发",
        "en": "Max Concurrent",
    },
    "gpu_load": {
        "zh": "GPU 负载分布:",
        "en": "GPU Load Distribution:",
    },
    "can_accept_task": {
        "zh": "可接受新任务 (剩余 {slots} 槽位)",
        "en": "Can accept new tasks ({slots} slots available)",
    },
    "queue_full": {
        "zh": "队列已满，新任务需排队等待",
        "en": "Queue full, new tasks will be queued",
    },
    "cannot_get_queue_status": {
        "zh": "无法获取队列状态: {error}",
        "en": "Cannot get queue status: {error}",
    },
    "language": {
        "zh": "语言",
        "en": "Language",
    },
    
    # ==========================================
    # 主界面
    # ==========================================
    "input_placeholder": {
        "zh": "请输入指令...",
        "en": "Enter your request...",
    },
    "thinking": {
        "zh": "正在思考和执行任务...",
        "en": "Thinking and executing...",
    },
    "no_chat": {
        "zh": "暂无对话，请在左侧点击 [+ 新建对话] 按钮。",
        "en": "No conversation. Click [+ New Chat] on the left.",
    },
    "api_key_invalid": {
        "zh": "Google API Key 无效。",
        "en": "Google API Key is invalid.",
    },
    "agent_init_failed": {
        "zh": "模型/Agent 初始化失败: {error}",
        "en": "Model/Agent initialization failed: {error}",
    },
    "execution_error": {
        "zh": "执行出错: {error}",
        "en": "Execution error: {error}",
    },
    
    # ==========================================
    # 可视化相关
    # ==========================================
    "structure_preview": {
        "zh": "结构预览",
        "en": "Structure Preview",
    },
    "analysis_results": {
        "zh": "分析结果可视化",
        "en": "Analysis Results Visualization",
    },
    "cluster_structure": {
        "zh": "团簇结构",
        "en": "Cluster Structure",
    },
    "homo_orbital": {
        "zh": "HOMO 轨道",
        "en": "HOMO Orbital",
    },
    "lumo_orbital": {
        "zh": "LUMO 轨道",
        "en": "LUMO Orbital",
    },
    "esp": {
        "zh": "静电势 (ESP)",
        "en": "Electrostatic Potential (ESP)",
    },
    "li_deformation": {
        "zh": "Li Deformation",
        "en": "Li Deformation",
    },
    "download_results": {
        "zh": "下载分析结果压缩包 (.zip)",
        "en": "Download Analysis Package (.zip)",
    },
    "view_3d_structure": {
        "zh": "点击查看 3D 分子结构 (可交互)",
        "en": "Click to view 3D molecular structure (interactive)",
    },
    "structure_path": {
        "zh": "结构路径",
        "en": "Structure path",
    },
    "continue_analysis_hint": {
        "zh": "提示：您可以说「对上面生成的结构进行电子结构分析」来继续分析",
        "en": "Tip: You can say 'Analyze the structure above' to continue with electronic structure analysis",
    },
    "more_structures_hint": {
        "zh": "还有 {count} 个团簇未显示。完整结果请下载分析包查看。",
        "en": "{count} more clusters not shown. Download the analysis package for complete results.",
    },
    "cannot_load_preview": {
        "zh": "无法加载结构预览",
        "en": "Cannot load structure preview",
    },
    "preview_failed": {
        "zh": "结构预览失败: {error}",
        "en": "Structure preview failed: {error}",
    },
    "file_not_exist": {
        "zh": "结构文件不存在: {path}",
        "en": "Structure file does not exist: {path}",
    },
    "total_clusters": {
        "zh": "共 {total} 个团簇，显示前 {displayed} 个",
        "en": "{total} clusters total, showing first {displayed}",
    },
    "formula": {
        "zh": "配方",
        "en": "Recipe",
    },
    "chemical_formula": {
        "zh": "化学式",
        "en": "Chemical Formula",
    },
    "atom_count": {
        "zh": "原子数",
        "en": "Atom Count",
    },
    "isovalue_settings": {
        "zh": "等值面设置",
        "en": "Isosurface Settings",
    },
    "isovalue_size": {
        "zh": "等值面大小",
        "en": "Isosurface Size",
    },
    "current_value": {
        "zh": "当前值",
        "en": "Current Value",
    },
    "isovalue_hint": {
        "zh": "调大：轨道包络面收缩；调小：轨道包络面扩展",
        "en": "Larger: orbital envelope contracts; Smaller: orbital envelope expands",
    },
    "esp_visualization": {
        "zh": "静电势 (ESP) 可视化",
        "en": "Electrostatic Potential (ESP) Visualization",
    },
    "esp_description": {
        "zh": "展示分子表面静电势分布：红色为正（亲核区域），蓝色为负（亲电区域）",
        "en": "Shows molecular surface electrostatic potential: red is positive (nucleophilic region), blue is negative (electrophilic region)",
    },
    "color_scale_range": {
        "zh": "色阶范围 (eV)",
        "en": "Color Scale Range (eV)",
    },
    "color_scale_hint": {
        "zh": "调整 ESP 色阶的显示范围，超出范围的值会被截断到边界颜色",
        "en": "Adjust ESP color scale range; values outside this range will be clamped to boundary colors",
    },
    "range": {
        "zh": "±范围",
        "en": "±Range",
    },
    "density_file": {
        "zh": "密度文件",
        "en": "Density File",
    },
    "esp_file": {
        "zh": "ESP文件",
        "en": "ESP File",
    },
    "esp_extrema_info": {
        "zh": "ESP 极值信息",
        "en": "ESP Extrema Information",
    },
    "max_value": {
        "zh": "最大值 (eV)",
        "en": "Maximum (eV)",
    },
    "min_value": {
        "zh": "最小值 (eV)",
        "en": "Minimum (eV)",
    },
    "location": {
        "zh": "位置",
        "en": "Location",
    },
    "li_deformation_visualization": {
        "zh": "Li 离子变形因子可视化",
        "en": "Li Ion Deformation Factor Visualization",
    },
    "li_deformation_description": {
        "zh": "展示 Li 离子周围电子密度变形的等值面分布",
        "en": "Shows the isosurface distribution of electron density deformation around Li ion",
    },
    "surface_opacity": {
        "zh": "表面透明度",
        "en": "Surface Opacity",
    },
    "opacity_hint": {
        "zh": "调整 Li deformation 表面的透明度",
        "en": "Adjust the opacity of Li deformation surface",
    },
    "opacity": {
        "zh": "透明度",
        "en": "Opacity",
    },
    "no_molecule_file": {
        "zh": "未找到分子结构文件，无法叠加显示",
        "en": "Molecule structure file not found, cannot overlay",
    },
    "file": {
        "zh": "文件",
        "en": "File",
    },
    "isovalue": {
        "zh": "等值面",
        "en": "Isovalue",
    },
    "homo_vis_failed": {
        "zh": "HOMO 可视化失败: {error}",
        "en": "HOMO visualization failed: {error}",
    },
    "lumo_vis_failed": {
        "zh": "LUMO 可视化失败: {error}",
        "en": "LUMO visualization failed: {error}",
    },
    "esp_vis_failed": {
        "zh": "ESP 可视化失败: {error}",
        "en": "ESP visualization failed: {error}",
    },
    "li_deform_vis_failed": {
        "zh": "Li Deformation 可视化失败: {error}",
        "en": "Li Deformation visualization failed: {error}",
    },
    
    # ==========================================
    # 可视化 HTML 内嵌文本
    # ==========================================
    "vis_mouse_hint": {
        "zh": "🖱️ 左键拖动旋转 | 滚轮缩放 | 右键平移",
        "en": "🖱️ Left-drag to rotate | Scroll to zoom | Right-drag to pan",
    },
    "vis_atom_count": {
        "zh": "原子数",
        "en": "Atoms",
    },
    "vis_orbital": {
        "zh": "轨道",
        "en": "Orbital",
    },
    "vis_isovalue": {
        "zh": "等值面",
        "en": "Isovalue",
    },
    "vis_positive_phase": {
        "zh": "正相位",
        "en": "Positive",
    },
    "vis_negative_phase": {
        "zh": "负相位",
        "en": "Negative",
    },
    "vis_deformation_region": {
        "zh": "变形区域",
        "en": "Deformation Region",
    },
    "vis_esp_label": {
        "zh": "静电势 (ESP)",
        "en": "Electrostatic Potential (ESP)",
    },
    "vis_density_isovalue": {
        "zh": "密度等值面",
        "en": "Density Isovalue",
    },
    "vis_red_positive": {
        "zh": "红=正(亲核)",
        "en": "Red=Positive(Nucleophilic)",
    },
    "vis_blue_negative": {
        "zh": "蓝=负(亲电)",
        "en": "Blue=Negative(Electrophilic)",
    },
    "vis_positive_nucleophilic": {
        "zh": "正(亲核)",
        "en": "Positive",
    },
    "vis_negative_electrophilic": {
        "zh": "负(亲电)",
        "en": "Negative",
    },
    "vis_missing_dep": {
        "zh": "缺少依赖包",
        "en": "Missing dependency",
    },
    "vis_run_pip": {
        "zh": "请运行",
        "en": "Please run",
    },
    "vis_load_orbital_failed": {
        "zh": "加载轨道可视化失败",
        "en": "Failed to load orbital visualization",
    },
    "vis_load_li_deform_failed": {
        "zh": "加载 Li Deformation 可视化失败",
        "en": "Failed to load Li Deformation visualization",
    },
    "vis_load_esp_failed": {
        "zh": "加载 ESP 可视化失败",
        "en": "Failed to load ESP visualization",
    },
    "vis_density_file_not_exist": {
        "zh": "密度 Cube 文件不存在",
        "en": "Density Cube file does not exist",
    },
    "vis_esp_file_not_exist": {
        "zh": "ESP Cube 文件不存在",
        "en": "ESP Cube file does not exist",
    },
    "vis_structure_not_exist": {
        "zh": "结构文件不存在",
        "en": "Structure file does not exist",
    },
    
    # ==========================================
    # 错误信息
    # ==========================================
    "error_path_not_found": {
        "zh": "错误：终止执行\n\n找不到{description}：`{path}`\n\n请检查文件路径是否正确。",
        "en": "Error: Execution terminated\n\nCannot find {description}: `{path}`\n\nPlease check if the file path is correct.",
    },
    "folder_delete_failed": {
        "zh": "文件夹删除失败: {error}",
        "en": "Folder deletion failed: {error}",
    },
}


# ==============================================================================
# 欢迎消息
# ==============================================================================

WELCOME_MESSAGES: dict[Language, str] = {
    "zh": """您好！我是 EMolAgent，您的计算化学 AI 助手。

我具备两大核心能力：

🔬 **分子团簇计算**
从本地数据库检索分子、构建并优化团簇结构，预测 HOMO/LUMO、偶极矩及静电势等电子性质。
示例：「请构建一个包含 1个Li离子、3个DME分子 和 1个FSI阴离子 的团簇」

📚 **文献知识问答**
基于数百篇 AI for Science 和电解液领域文献，回答相关学术问题。
示例：「什么是溶剂化结构？CIP和SSIP有什么区别？」「介绍一下 GNN 在分子性质预测中的应用」

请告诉我您的需求，我将为您提供帮助！""",
    
    "en": """Hello! I'm EMolAgent, your computational chemistry AI assistant.

I have two core capabilities:

🔬 **Molecular Cluster Computation**
Retrieve molecules from a local database, build and optimize cluster structures, and predict electronic properties such as HOMO/LUMO, dipole moments, and electrostatic potentials.
Example: "Please build a cluster containing 1 Li ion, 3 DME molecules, and 1 FSI anion"

📚 **Literature Knowledge Q&A**
Answer academic questions based on hundreds of AI for Science and electrolyte-related papers.
Example: "What is solvation structure? What's the difference between CIP and SSIP?" "Introduce the application of GNN in molecular property prediction"

Please tell me your needs and I'll be happy to help!""",
}


# ==============================================================================
# 系统提示词
# ==============================================================================

SYSTEM_PROMPTS: dict[Language, str] = {
    "zh": """
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
""",

    "en": """
You are EMolAgent, a computational chemistry AI assistant. You have two core capabilities:

## Capability 1: Molecular Cluster Computation
Follow this workflow to handle user's molecular computation requests:

### Important: Identify User Intent
User intent can be categorized as follows:

1. **Generate structure only (no analysis)**:
   - Keywords: "generate structure", "build cluster", "create molecule"
   - **Does NOT include**: "analyze", "electronic structure", "predict properties", "HOMO", "LUMO"
   - Action: Only call build tools, **DO NOT call** `Run_Inference_Pipeline`

2. **Generate and analyze (complete workflow)**:
   - Keywords: "generate and analyze", "compute electronic structure", "predict properties", "perform analysis"
   - Action: Call build tools + `Run_Inference_Pipeline`

3. **Analyze existing structure**:
   - Keywords: "analyze the structure above", "analyze previous structure", "continue analysis"
   - Action: Find `optimized_db` path from conversation history, only call `Run_Inference_Pipeline`

### ⚠️ Key: Single Recipe vs Multiple Recipes
- **Single recipe**: User describes only one recipe (e.g., "1Li+3DME+1FSI")
  - Generate only: `Build_Structure_Only`
  - Generate and analyze: `Build_and_Optimize` + `Run_Inference_Pipeline`

- **Multiple recipes**: User describes different recipes (e.g., "build recipe A... then build recipe B...")
  - **Must** use `Build_Multiple_Clusters` to process all recipes at once
  - Generate only: `Build_Multiple_Clusters` (don't call Run_Inference_Pipeline)
  - Generate and analyze: `Build_Multiple_Clusters` + `Run_Inference_Pipeline`

### Workflow Steps:

1.  **Parse requirements**:
    * Identify center ion, solvents, anions and their quantities
    * Determine if single or multiple recipes
    * **Key judgment**: Does user request electronic structure analysis?

2.  **Database search (Search_Molecule_DB)**:
    * For each molecule (solvent or anion), call `Search_Molecule_DB` first
    * Search 'solvent' type for Solvent, 'anion' type for Anion
    * If `db_path` found, tell user "Found XXX in database"
    * If not found, use SMILES

3.  **Modeling and optimization**:
    * Single recipe, generate only: `Build_Structure_Only`
    * Single recipe, full analysis: `Build_and_Optimize` → `Run_Inference_Pipeline`
    * Multiple recipes, generate only: `Build_Multiple_Clusters` (**stop here, don't call analysis**)
    * Multiple recipes, full analysis: `Build_Multiple_Clusters` → `Run_Inference_Pipeline`

4.  **Electronic structure inference (Run_Inference_Pipeline)**:
    * **Only call when user explicitly requests analysis**
    * Use `optimized_db` path from previous step
    * For multiple recipes, only call once, will process all structures

5.  **Final report**:
    * Generate only: Show 3D preview, hint user can analyze later
    * With analysis: Show electronic properties, keep `[[DOWNLOAD:...]]` link

### Remember:
- User didn't say "analyze" or "electronic structure" → Don't call `Run_Inference_Pipeline`
- User says "analyze the structure above" → Find `optimized_db` from history, call `Run_Inference_Pipeline`
- Multiple recipes = Use `Build_Multiple_Clusters`, don't call single recipe tools multiple times

## Capability 2: Literature Knowledge Q&A (Search_Knowledge_Base)
Use `Search_Knowledge_Base` tool when user asks about:
- AI for Science models and methods (GNN, Transformer, diffusion models, etc.)
- Electrolyte properties, solvation structures, ion transport mechanisms
- Battery materials, lithium-ion/sodium-ion batteries
- Molecular simulation methods, DFT calculations, machine learning potentials
- Any scientific concept requiring literature support

**Knowledge Q&A workflow**:
1. Understand core concepts in user's question
2. Call `Search_Knowledge_Base` to search relevant literature
3. Based on retrieved content, combine with your knowledge for comprehensive answer
4. **Must cite sources**, format: "According to [xxx.pdf]..."

【Notes】
* If user says "3 DME", it means count=3.
* FSI is usually an anion.
* Execute step by step, don't skip "database search" as database structures have highest quality.
* For knowledge questions, prioritize knowledge base search to ensure literature-backed answers.
""",
}


# ==============================================================================
# 辅助函数
# ==============================================================================

def get_text(key: str, lang: Language = "zh") -> str:
    """
    获取指定 key 的翻译文本。
    
    Args:
        key: 翻译键
        lang: 语言代码 ("zh" 或 "en")
    
    Returns:
        翻译后的文本，如果 key 不存在则返回 key 本身
    """
    if key in TRANSLATIONS:
        return TRANSLATIONS[key].get(lang, TRANSLATIONS[key].get("zh", key))
    return key


def t(key: str, lang: Language = "zh", **kwargs) -> str:
    """
    获取翻译文本的简写函数，支持格式化参数。
    
    Args:
        key: 翻译键
        lang: 语言代码
        **kwargs: 格式化参数
    
    Returns:
        翻译并格式化后的文本
    
    Example:
        t("index_complete", "zh", new=10, skipped=5, failed=0)
        # 返回: "索引完成！新增: 10, 跳过: 5, 失败: 0"
    """
    text = get_text(key, lang)
    if kwargs:
        try:
            return text.format(**kwargs)
        except KeyError:
            return text
    return text


def get_welcome_message(lang: Language = "zh") -> str:
    """
    获取指定语言的欢迎消息。
    
    Args:
        lang: 语言代码
    
    Returns:
        欢迎消息文本
    """
    return WELCOME_MESSAGES.get(lang, WELCOME_MESSAGES["zh"])


def get_system_prompt(lang: Language = "zh") -> str:
    """
    获取指定语言的系统提示词。
    
    Args:
        lang: 语言代码
    
    Returns:
        系统提示词文本
    """
    return SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["zh"])
