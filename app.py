# app.py (Main Streamlit application file)

# Python Standard Library Imports
import logging
import os
import sys 
import traceback

# Third-party Imports
import streamlit as st # Import Streamlit first

# --- 1. Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="AI小说写作助手",
    page_icon="✍️", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your_repo/issues', # Replace with your actual help/issue URL
        'Report a bug': "https://github.com/your_repo/issues", # Replace
        'About': """
        ## AI 小说写作助手
        这是一个利用AI辅助小说创作的工具。
        技术栈: Streamlit, Milvus (Zilliz Cloud), Sentence Transformers, LLMs.
        """
    }
)

# --- 2. Attempt to Import Core Logic (novel_core.py) ---
_CORE_IMPORTED_SUCCESSFULLY = False
_CORE_IMPORT_ERROR_MESSAGE = ""
_CORE_IMPORT_ERROR_TRACEBACK = ""
try:
    # Configure basic logging for app_ui BEFORE importing novel_core
    ui_logger = logging.getLogger("AppUI") 
    if not ui_logger.hasHandlers(): # Avoid duplicate handlers if script re-runs
        logging.basicConfig(level=logging.DEBUG, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ui_logger.info("app.py: Script execution started. Attempting to import novel_core...")
    
    import novel_core 
    _CORE_IMPORTED_SUCCESSFULLY = True
    ui_logger.info("app.py: novel_core imported successfully!")

except ImportError as e_imp:
    _CORE_IMPORT_ERROR_MESSAGE = f"CRITICAL ERROR: 无法导入 novel_core.py。请确保该文件与 app.py 在同一目录下，并且所有依赖（如openai, pymilvus等）已在requirements.txt中声明。错误: {e_imp}"
    _CORE_IMPORT_ERROR_TRACEBACK = traceback.format_exc()
    if 'ui_logger' in globals(): ui_logger.critical(_CORE_IMPORT_ERROR_MESSAGE, exc_info=False) # exc_info=False as traceback is already formatted
    else: print(_CORE_IMPORT_ERROR_MESSAGE + "\n" + _CORE_IMPORT_ERROR_TRACEBACK)
except Exception as e_core_load_time: 
    _CORE_IMPORT_ERROR_MESSAGE = f"CRITICAL ERROR: 导入或执行 novel_core.py 顶层代码时发生意外错误: {e_core_load_time}"
    _CORE_IMPORT_ERROR_TRACEBACK = traceback.format_exc()
    if 'ui_logger' in globals(): ui_logger.critical(_CORE_IMPORT_ERROR_MESSAGE, exc_info=False)
    else: print(_CORE_IMPORT_ERROR_MESSAGE + "\n" + _CORE_IMPORT_ERROR_TRACEBACK)

# --- 3. Define Placeholders if Core Logic Failed to Import ---
if not _CORE_IMPORTED_SUCCESSFULLY:
    # This class provides default values if novel_core.py cannot be imported,
    # allowing the Streamlit UI to at least render an error message.
    class NovelCorePlaceholder:
        embedding_providers_map_core = {"0": ("错误", "错误", "核心模块加载失败", 0)}
        llm_providers_map_core = {"0": "错误"}
        NOVEL_MD_OUTPUT_DIR_CORE = "./novel_markdown_placeholder"
        # Add stubs for functions that might be called during session state init if needed
        def core_initialize_system(self, *args, **kwargs): 
            raise RuntimeError("novel_core未加载，无法初始化。")
        def core_generate_segment_text_for_ui(self, *args, **kwargs): return "novel_core未加载。"
        def core_adopt_segment_from_ui(self, *args, **kwargs): return False
        # Add any other constants app.py's initialize_session_state_ui might directly access
        OPENAI_API_KEY_ENV_NAME="OPENAI_API_KEY_PLACEHOLDER" # Example
        GEMINI_API_KEY_ENV_NAME="GEMINI_API_KEY_PLACEHOLDER" # Example
    novel_core = NovelCorePlaceholder() # type: ignore


# --- 4. Initialize Streamlit Session State ---
def initialize_session_state_ui():
    # Uses novel_core constants for defaults, with fallbacks if novel_core failed to import
    emb_map = novel_core.embedding_providers_map_core
    llm_map = novel_core.llm_providers_map_core
    md_dir = novel_core.NOVEL_MD_OUTPUT_DIR_CORE
    
    default_emb_key = list(emb_map.keys())[1] if len(emb_map.keys()) > 1 else (list(emb_map.keys())[0] if emb_map else "0")
    default_llm_key = list(llm_map.keys())[2] if len(llm_map.keys()) > 2 else (list(llm_map.keys())[0] if llm_map else "0")

    defaults = {
        "system_initialized_attempted": False,
        "system_initialized_successfully": False, 
        "log_messages": ["UI应用已启动，等待初始化..."],
        "current_chapter": 1,
        "current_segment_number": 0, 
        "current_generated_text": "", 
        "user_directive_for_current_segment": "",
        "selected_embedding_provider_key": default_emb_key,
        "selected_llm_provider_key": default_llm_key,
        "api_keys": {}, 
        "max_tokens_per_llm_call": int(os.getenv("MAX_TOKENS_PER_LLM_CALL", "7800")),
        "segments_per_chapter_advance": int(os.getenv("SEGMENTS_PER_CHAPTER_ADVANCE", "3")),
        "last_adopted_segment_text": "这是故事的开端，尚无先前的故事片段。",
        "user_directive_for_current_segment_buffer": "", 
        "show_expand_input": False, 
        "num_recent_segments_to_fetch_ui": 2, 
        "llm_temperature": 0.7, 
        "novel_md_output_dir_ui": md_dir,
        "last_known_chapter": None, 
        "last_known_segment": None, 
        "resume_choice_made": False,
        "resume_choice_idx": 0,
        # Keys that novel_core.py will set, good to have them defined
        "selected_embedding_provider_identifier": None,
        "embedding_dimension": None,
        "selected_st_model_name": None,
        "current_llm_provider": None,
        "embedding_model_instance": None,
        "lore_collection_milvus_obj": None,
        "story_collection_milvus_obj": None,
        "milvus_initialized_core": False,
        "lore_collection_name": "未初始化",
        "story_collection_name": "未初始化",
        "milvus_target": "未知"
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

initialize_session_state_ui() # Call it once at the start of the UI script

# In app.py
def add_log_ui(message: str, level: str = "info", UImodule: str = "UI"):
    console_logger_ui = logging.getLogger("AppUI") 
    
    # --- MODIFIED LOGIC FOR llm_provider_for_log ---
    llm_provider_for_log_val = "SYS_UNKNOWN" # A safe default string

    if _CORE_IMPORTED_SUCCESSFULLY and hasattr(st, 'session_state'): # Check if session_state exists
        if st.session_state.get('system_initialized_successfully'):
            llm_provider_for_log_val = st.session_state.get('current_llm_provider', 'SYS_INIT_OK')
        elif st.session_state.get('system_initialized_attempted'):
            # Get the key first, then the name from map
            selected_key = st.session_state.get('selected_llm_provider_key')
            if selected_key and selected_key in novel_core.llm_providers_map_core:
                llm_provider_for_log_val = novel_core.llm_providers_map_core[selected_key]
            else:
                llm_provider_for_log_val = 'SYS_INIT_ATTEMPT_NO_KEY'
        else: # Not attempted, or core not imported successfully enough to have maps
            llm_provider_for_log_val = "SYS_PRE_INIT"
            
    # Ensure llm_provider_for_log_val is a string before .upper()
    if not isinstance(llm_provider_for_log_val, str):
        llm_provider_for_log_val = str(llm_provider_for_log_val) # Convert to string if it became None or other type
    # --- END MODIFICATION ---

    timestamp = f"[{UImodule}][{llm_provider_for_log_val.upper()}] " # Now this should be safe
    log_entry = f"{timestamp}{message}"
    
    # Ensure log_messages exists and is a list
    if 'log_messages' not in st.session_state or not isinstance(st.session_state.log_messages, list): 
        st.session_state.log_messages = ["Log list init error in add_log_ui."]
    st.session_state.log_messages.insert(0, log_entry)
    
    # Console logging
    if level.lower() == "info": console_logger_ui.info(message)
    # ... (other levels) ...
    elif level.lower() == "warning": console_logger_ui.warning(message)
    elif level.lower() == "error": console_logger_ui.error(message)
    elif level.lower() == "fatal": console_logger_ui.critical(message) 
    elif level.lower() == "debug": console_logger_ui.debug(message)
    elif level.lower() == "success": console_logger_ui.info(f"SUCCESS: {message}")


# --- 5. Display Critical Import Error if Occurred (AFTER page_config and session_state init) ---
if not _CORE_IMPORTED_SUCCESSFULLY:
    st.title("AI 小说写作助手 - 启动失败") # Title can be set even if page_config failed due to this
    st.error(_CORE_IMPORT_ERROR_MESSAGE if _CORE_IMPORT_ERROR_MESSAGE else "无法加载核心模块 novel_core.py")
    if _CORE_IMPORT_ERROR_TRACEBACK:
        st.subheader("错误详情 (Traceback):")
        st.code(_CORE_IMPORT_ERROR_TRACEBACK)
    st.warning("请检查应用的日志（在Hugging Face Spaces的Logs标签页，或本地运行时的控制台）获取更多信息。确保所有依赖已在 `requirements.txt` 中正确声明并安装成功。")
    st.stop() # Stop further UI rendering if core module is missing

# --- UI Layout (Assuming _CORE_IMPORTED_SUCCESSFULLY is True from here) ---
st.title("✍️ AI 小说写作助手")

with st.sidebar:
    st.header("系统配置")

    # Embedding Provider Selection
    emb_map_ui_sidebar = novel_core.embedding_providers_map_core
    emb_keys_ui_sidebar = list(emb_map_ui_sidebar.keys())
    current_selected_emb_key_sidebar = st.session_state.selected_embedding_provider_key
    if current_selected_emb_key_sidebar not in emb_keys_ui_sidebar: # Fallback if key invalid
        current_selected_emb_key_sidebar = emb_keys_ui_sidebar[0] if emb_keys_ui_sidebar else None
        st.session_state.selected_embedding_provider_key = current_selected_emb_key_sidebar
    default_emb_idx_sidebar = emb_keys_ui_sidebar.index(current_selected_emb_key_sidebar) if current_selected_emb_key_sidebar and emb_keys_ui_sidebar else 0
    
    selected_emb_key_widget = st.selectbox(
        "选择嵌入模型", options=emb_keys_ui_sidebar,
        format_func=lambda x: f"{x}. {emb_map_ui_sidebar[x][0].upper()} {emb_map_ui_sidebar[x][2]}" if x in emb_map_ui_sidebar else "错误选项",
        index=default_emb_idx_sidebar, key="emb_provider_selector_final",
        disabled=st.session_state.system_initialized_attempted )
    if selected_emb_key_widget != st.session_state.selected_embedding_provider_key: # Update on change
        st.session_state.selected_embedding_provider_key = selected_emb_key_widget

    # LLM Provider Selection
    llm_map_ui_sidebar = novel_core.llm_providers_map_core
    llm_keys_ui_sidebar = list(llm_map_ui_sidebar.keys())
    current_selected_llm_key_sidebar = st.session_state.selected_llm_provider_key
    if current_selected_llm_key_sidebar not in llm_keys_ui_sidebar:
        current_selected_llm_key_sidebar = llm_keys_ui_sidebar[0] if llm_keys_ui_sidebar else None
        st.session_state.selected_llm_provider_key = current_selected_llm_key_sidebar
    default_llm_idx_sidebar = llm_keys_ui_sidebar.index(current_selected_llm_key_sidebar) if current_selected_llm_key_sidebar and llm_keys_ui_sidebar else 0

    selected_llm_key_widget = st.selectbox(
        "选择LLM模型", options=llm_keys_ui_sidebar,
        format_func=lambda x: f"{x}. {llm_map_ui_sidebar[x].upper()}" if x in llm_map_ui_sidebar else "错误选项",
        index=default_llm_idx_sidebar, key="llm_provider_selector_final",
        disabled=st.session_state.system_initialized_attempted )
    if selected_llm_key_widget != st.session_state.selected_llm_provider_key:
        st.session_state.selected_llm_provider_key = selected_llm_key_widget
    
    with st.expander("高级配置/API Keys (可选)", expanded=False):
        # API Key inputs using novel_core constants for keys
        st.session_state.api_keys[novel_core.OPENAI_API_KEY_ENV_NAME] = st.text_input( "OpenAI API Key", value=st.session_state.api_keys.get(novel_core.OPENAI_API_KEY_ENV_NAME, os.getenv(novel_core.OPENAI_API_KEY_ENV_NAME,"")), type="password", key="openai_key_input_final", disabled=st.session_state.system_initialized_attempted)
        st.session_state.api_keys[novel_core.GEMINI_API_KEY_ENV_NAME] = st.text_input( "Gemini API Key", value=st.session_state.api_keys.get(novel_core.GEMINI_API_KEY_ENV_NAME, os.getenv(novel_core.GEMINI_API_KEY_ENV_NAME,"")), type="password", key="gemini_key_input_final", disabled=st.session_state.system_initialized_attempted)
        # TODO: Add inputs for DEEPSEEK_API_KEY_ENV_NAME and CUSTOM_PROXY_API_KEY_ENV_NAME if needed

        st.session_state.max_tokens_per_llm_call = st.number_input("LLM单次最大Token", min_value=200, max_value=32000, value=st.session_state.max_tokens_per_llm_call, step=100, key="max_tokens_final")
        st.session_state.llm_temperature = st.slider("LLM Temperature", 0.0, 2.0, st.session_state.llm_temperature, 0.05, key="temp_final")
        st.session_state.segments_per_chapter_advance = st.number_input("每章节片段数提示进阶", min_value=1, max_value=20, value=st.session_state.segments_per_chapter_advance, step=1, key="segments_chap_final")
        st.session_state.num_recent_segments_to_fetch_ui = st.number_input("检索最近故事上下文数量", min_value=0, max_value=5, value=st.session_state.num_recent_segments_to_fetch_ui, step=1, key="recent_segs_final")
        st.session_state.novel_md_output_dir_ui = st.text_input("Markdown输出目录 (本地)", value=st.session_state.novel_md_output_dir_ui, key="md_dir_final")

    # Initialization Button Logic
    if not st.session_state.system_initialized_attempted:
        if st.sidebar.button("🚀 初始化系统", key="init_button_final_v4_ui", type="primary", use_container_width=True):
            st.session_state.system_initialized_attempted = True # Mark that init process has been clicked
            with st.spinner("系统初始化中... (可能需要较长时间下载模型或连接服务)"):
                try:
                    add_log_ui(f"UI 发起初始化。选择的嵌入Key: '{st.session_state.selected_embedding_provider_key}', LLM Key: '{st.session_state.selected_llm_provider_key}'")
                    novel_core.core_initialize_system( # Call the main init function in novel_core
                        st.session_state.selected_embedding_provider_key,
                        st.session_state.selected_llm_provider_key,
                        st.session_state.api_keys 
                    ) # This function should set st.session_state.system_initialized_successfully
                except Exception as e_init_call_ui: 
                    st.session_state.system_initialized_successfully = False # Ensure this is set on failure
                    add_log_ui(f"UI层面捕获到初始化失败: {e_init_call_ui}", "error")
    elif st.session_state.system_initialized_successfully:
        llm_disp = novel_core.llm_providers_map_core.get(st.session_state.selected_llm_provider_key, '未知').upper()
        emb_disp = novel_core.embedding_providers_map_core.get(st.session_state.selected_embedding_provider_key, ["未知"])[0].upper()
        st.sidebar.success(f"✅ 系统已初始化！\nLLM: {llm_disp}\nEmbedding: {emb_disp}")
        if st.sidebar.button("🔄 重新初始化/切换模型", key="reinit_button_final_ui", use_container_width=True):
            # Reset states to allow re-initialization
            st.session_state.system_initialized_attempted = False
            st.session_state.system_initialized_successfully = False
            # Reset core-specific flags that might affect re-init logic in novel_core
            if _CORE_IMPORTED_SUCCESSFULLY: # Only if novel_core was available
                st.session_state.milvus_initialized_core = False 
                st.session_state.loaded_st_model_name = None
                st.session_state.embedding_model_instance = None
                st.session_state.llm_client = None
                st.session_state.gemini_llm_client_core = None
            st.session_state.current_chapter = 1 
            st.session_state.current_segment_number = 0
            st.session_state.current_generated_text = ""
            st.session_state.user_directive_for_current_segment_buffer = ""
            st.session_state.resume_choice_made = False 
            st.session_state.last_known_chapter = None 
            st.session_state.last_known_segment = None
            add_log_ui("用户请求重新初始化系统。")
    # If attempted but not successful, main area will show the error below

# --- Main Writing Area ---
if st.session_state.system_initialized_attempted:
    if st.session_state.system_initialized_successfully:
        st.header("📝 小说创作区")

        # Resume logic UI part
        if st.session_state.get('last_known_chapter') is not None and \
           st.session_state.get('last_known_segment') is not None and \
           not st.session_state.resume_choice_made:
            # ... (Full radio button and confirm button logic for resume choice from previous version) ...
            pass # Placeholder for brevity
        
        # --- Fix for TypeError on display ---
        current_seg_num_for_display = st.session_state.get('current_segment_number', 0)
        if not isinstance(current_seg_num_for_display, int):
            add_log_ui(f"警告: current_segment_number Type ({type(current_seg_num_for_display).__name__}) invalid. Reset to 0.", "warning")
            current_seg_num_for_display = 0; st.session_state.current_segment_number = 0
        current_chap_for_display = st.session_state.get('current_chapter', 1)
        if not isinstance(current_chap_for_display, int):
            add_log_ui(f"警告: current_chapter Type ({type(current_chap_for_display).__name__}) invalid. Reset to 1.", "warning")
            current_chap_for_display = 1; st.session_state.current_chapter = 1
        st.info(f"当前写作进度：章节 {current_chap_for_display}, 计划生成片段号 {current_seg_num_for_display + 1}")
        
        # --- TODO: PASTE YOUR FULL MAIN WRITING AREA UI HERE ---
        # This includes: Directive input, Generate button, Display Area, Action buttons.
        # Ensure their callbacks call the correct novel_core.core_..._for_ui functions.
        st.session_state.user_directive_for_current_segment = st.text_area(
            "写作指令:", height=200, 
            value=st.session_state.user_directive_for_current_segment, 
            key="main_directive_input_final"
        )
        if st.button("✨ 生成故事片段", key="main_generate_button_final", type="primary"):
            if st.session_state.user_directive_for_current_segment.strip():
                # ... (Spinner, call novel_core.core_generate_segment_text_for_ui, update session_state) ...
                add_log_ui("生成片段按钮点击 (实际生成逻辑待填充 novel_core)。")
                st.session_state.current_generated_text = novel_core.core_generate_segment_text_for_ui(st.session_state.user_directive_for_current_segment)
                st.session_state.user_directive_for_current_segment_buffer = st.session_state.user_directive_for_current_segment
            else:
                st.warning("写作指令不能为空。")
        
        if st.session_state.current_generated_text:
            st.markdown("---"); st.subheader("🤖 AI 生成的片段"); 
            st.markdown(st.session_state.current_generated_text)
            st.caption(f"字数: {len(st.session_state.current_generated_text)}")
            # TODO: Add action buttons (Adopt, Rewrite, Expand, Discard, Next) and their callbacks here
            # Example: if st.button("👍 采纳"): novel_core.core_adopt_segment_from_ui(...)

    else: # system_initialized_attempted is True, but system_initialized_successfully is False
        st.error("🤷 系统初始化失败，请检查侧边栏或下方的日志，并尝试在侧边栏重新初始化。")
elif not st.session_state.system_initialized_attempted and _CORE_IMPORTED_SUCCESSFULLY: 
    st.warning("👈 请在侧边栏选择模型并点击“初始化系统”以开始使用。")
elif not _CORE_IMPORTED_SUCCESSFULLY: # This case is handled at the top by st.error and st.stop()
    pass 


# --- Log Display Area (at the bottom of the main page) ---
st.markdown("---")
st.subheader("运行日志 (最新在前)")
log_container_main = st.container()
# ... (Full log display HTML/Markdown code from your previous complete app_ui.py) ...
log_html_content_main = "<div style='max-height: 300px; ...'>" # Placeholder for actual HTML
# ... (Loop through st.session_state.log_messages and build HTML) ...
log_container_main.markdown(log_html_content_main, unsafe_allow_html=True)

if st.button("清除主界面日志", key="clear_main_logs_button_final"):
    st.session_state.log_messages = ["UI主界面日志已清除。"]

add_log_ui("app.py: 脚本渲染到达末尾。", "debug")