# app_ui.py
import streamlit as st
import logging
import os
import sys # For sys.modules check if needed, though direct import check is usually enough
import traceback

# --- Attempt to Import Core Logic ---
_CORE_IMPORTED_SUCCESSFULLY = False
try:
    import novel_core 
    _CORE_IMPORTED_SUCCESSFULLY = True
except ImportError as e:
    # This error is critical and will be displayed by Streamlit if novel_core.py is missing
    # or has unrecoverable errors at import time (e.g., syntax errors in novel_core.py).
    st.error(f"CRITICAL ERROR: 无法导入 novel_core.py。请确保该文件与 app_ui.py 在同一目录下，"
             f"并且 novel_core.py 本身没有导致导入失败的严重错误。错误: {e}")
    # Add a placeholder for novel_core if it fails to import, to prevent further NameErrors with novel_core.CONSTANT
    class NovelCorePlaceholder:
        embedding_providers_map_core = {"0": ("错误", "错误", "核心模块加载失败", 0)}
        llm_providers_map_core = {"0": "错误"}
        NOVEL_MD_OUTPUT_DIR_CORE = "./novel_markdown_placeholder"
    novel_core = NovelCorePlaceholder()
    # st.stop() # Optionally stop execution, or let it run to show more UI context for the error.
except Exception as e_core_load_time: # Catch other errors during novel_core's initial import/load (e.g. top-level code in novel_core fails)
    st.error(f"CRITICAL ERROR: 导入或执行 novel_core.py 顶层代码时发生意外错误: {e_core_load_time}")
    st.code(traceback.format_exc())
    # Define placeholders if novel_core itself is problematic
    if 'novel_core' not in globals():
        class NovelCorePlaceholder:
            embedding_providers_map_core = {"0": ("错误", "错误", "核心模块加载失败", 0)}
            llm_providers_map_core = {"0": "错误"}
            NOVEL_MD_OUTPUT_DIR_CORE = "./novel_markdown_placeholder"
        novel_core = NovelCorePlaceholder()
    # st.stop()


# --- Page Configuration (set this early) ---
st.set_page_config(
    page_title="AI小说写作助手",
    page_icon="✍️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
def initialize_session_state_ui():
    # Use try-except here in case novel_core itself failed to import, providing safe defaults
    try:
        emb_map = novel_core.embedding_providers_map_core
        llm_map = novel_core.llm_providers_map_core
        md_dir = novel_core.NOVEL_MD_OUTPUT_DIR_CORE
        default_emb_key = list(emb_map.keys())[1] if len(emb_map.keys()) > 1 else (list(emb_map.keys())[0] if emb_map else "0")
        default_llm_key = list(llm_map.keys())[2] if len(llm_map.keys()) > 2 else (list(llm_map.keys())[0] if llm_map else "0")
    except Exception: # Fallback if novel_core or its constants are not available
        emb_map = {"0": ("错误", "错误", "核心模块加载失败", 0)}
        llm_map = {"0": "错误"}
        md_dir = "./novel_markdown_placeholder"
        default_emb_key = "0"
        default_llm_key = "0"

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
        "resume_choice_idx": 0 
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state_ui() # Call it once at the start

# --- Helper function to add logs to UI and console logger ---
def add_log_ui(message: str, level: str = "info", UImodule: str = "UI"):
    console_logger = logging.getLogger("AppUI") # Logger for console output from UI
    if not console_logger.hasHandlers(): # Basic config for UI logger if not set
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    timestamp = f"[{UImodule}][{st.session_state.get('current_llm_provider', 'SYS')}] "
    log_entry = f"{timestamp}{message}"
    
    if not isinstance(st.session_state.get("log_messages"), list): 
        st.session_state.log_messages = ["Log list init error."]
    st.session_state.log_messages.insert(0, log_entry) # For UI display
    
    # Log to console
    if level.lower() == "info": console_logger.info(message)
    elif level.lower() == "warning": console_logger.warning(message)
    elif level.lower() == "error": console_logger.error(message)
    elif level.lower() == "fatal": console_logger.critical(message) 
    elif level.lower() == "debug": console_logger.debug(message)
    elif level.lower() == "success": console_logger.info(f"SUCCESS: {message}")


# --- UI Layout ---
st.title("✍️ AI 小说写作助手")

with st.sidebar:
    st.header("系统配置")

    # Embedding Provider Selection
    emb_map_ui = novel_core.embedding_providers_map_core if _CORE_IMPORTED_SUCCESSFULLY else {"0": ("错误", "错误", "核心模块加载失败", 0)}
    emb_keys_ui = list(emb_map_ui.keys())
    # Ensure default_emb_key is valid for the current map
    current_selected_emb_key_ui = st.session_state.selected_embedding_provider_key
    if current_selected_emb_key_ui not in emb_keys_ui:
        current_selected_emb_key_ui = emb_keys_ui[0] if emb_keys_ui else None
        st.session_state.selected_embedding_provider_key = current_selected_emb_key_ui

    default_emb_idx_ui = emb_keys_ui.index(current_selected_emb_key_ui) if current_selected_emb_key_ui and emb_keys_ui else 0
    
    selected_emb_key_from_widget = st.selectbox(
        "选择嵌入模型",
        options=emb_keys_ui,
        format_func=lambda x: f"{x}. {emb_map_ui[x][0].upper()} {emb_map_ui[x][2]}" if x in emb_map_ui else "错误选项",
        index=default_emb_idx_ui,
        key="emb_provider_selector_ui_final",
        disabled=st.session_state.system_initialized_attempted 
    )
    if selected_emb_key_from_widget != st.session_state.selected_embedding_provider_key :
        st.session_state.selected_embedding_provider_key = selected_emb_key_from_widget
        # Rerun is implicit on selectbox change

    # LLM Provider Selection
    llm_map_ui = novel_core.llm_providers_map_core if _CORE_IMPORTED_SUCCESSFULLY else {"0": "错误"}
    llm_keys_ui = list(llm_map_ui.keys())
    current_selected_llm_key_ui = st.session_state.selected_llm_provider_key
    if current_selected_llm_key_ui not in llm_keys_ui:
        current_selected_llm_key_ui = llm_keys_ui[0] if llm_keys_ui else None
        st.session_state.selected_llm_provider_key = current_selected_llm_key_ui
        
    default_llm_idx_ui = llm_keys_ui.index(current_selected_llm_key_ui) if current_selected_llm_key_ui and llm_keys_ui else 0

    selected_llm_key_from_widget = st.selectbox(
        "选择LLM模型",
        options=llm_keys_ui,
        format_func=lambda x: f"{x}. {llm_map_ui[x].upper()}" if x in llm_map_ui else "错误选项",
        index=default_llm_idx_ui,
        key="llm_provider_selector_ui_final",
        disabled=st.session_state.system_initialized_attempted
    )
    if selected_llm_key_from_widget != st.session_state.selected_llm_provider_key:
        st.session_state.selected_llm_provider_key = selected_llm_key_from_widget

    with st.expander("高级配置/API Keys (可选)"):
        # API Key inputs using novel_core constants for keys
        if _CORE_IMPORTED_SUCCESSFULLY:
            st.session_state.api_keys[novel_core.OPENAI_API_KEY_ENV_NAME] = st.text_input(
                "OpenAI API Key", value=st.session_state.api_keys.get(novel_core.OPENAI_API_KEY_ENV_NAME, os.getenv(novel_core.OPENAI_API_KEY_ENV_NAME,"")), 
                type="password", key="openai_key_input_ui", disabled=st.session_state.system_initialized_attempted
            )
            st.session_state.api_keys[novel_core.GEMINI_API_KEY_ENV_NAME] = st.text_input(
                "Gemini API Key", value=st.session_state.api_keys.get(novel_core.GEMINI_API_KEY_ENV_NAME, os.getenv(novel_core.GEMINI_API_KEY_ENV_NAME,"")), 
                type="password", key="gemini_key_input_ui", disabled=st.session_state.system_initialized_attempted
            )
        # ... (Other advanced config inputs: max_tokens, temperature, etc.)
        st.session_state.max_tokens_per_llm_call = st.number_input("LLM单次最大Token", value=st.session_state.max_tokens_per_llm_call, key="max_tokens_ui")
        st.session_state.llm_temperature = st.slider("LLM Temperature", 0.0, 2.0, st.session_state.llm_temperature, 0.05, key="temp_ui")


    # Initialization Button Logic
    if not st.session_state.system_initialized_attempted:
        if st.sidebar.button("🚀 初始化系统", key="init_button_final_ui", type="primary", use_container_width=True,
                             disabled=not _CORE_IMPORTED_SUCCESSFULLY): # Disable if core didn't import
            if not _CORE_IMPORTED_SUCCESSFULLY:
                st.sidebar.error("核心模块未能加载，无法初始化！")
            else:
                st.session_state.system_initialized_attempted = True
                with st.spinner("系统初始化中... (可能需要较长时间)"):
                    try:
                        add_log_ui(f"UI 发起初始化。选择的嵌入Key: '{st.session_state.selected_embedding_provider_key}', LLM Key: '{st.session_state.selected_llm_provider_key}'")
                        novel_core.core_initialize_system( # Call the main init function in novel_core
                            st.session_state.selected_embedding_provider_key,
                            st.session_state.selected_llm_provider_key,
                            st.session_state.api_keys 
                        )
                        # novel_core.core_initialize_system should set "system_initialized_successfully"
                        # and log its own success/failure to st.session_state.log_messages
                    except Exception as e_init_call: # Catch errors from core_initialize_system itself
                        # This ensures system_initialized_successfully is False if core raises an unhandled error
                        st.session_state.system_initialized_successfully = False 
                        add_log_ui(f"调用核心初始化时发生严重错误: {e_init_call}", "FATAL")
                        st.sidebar.error(f"初始化调用失败: {str(e_init_call)[:100]}...") # Show in sidebar too
            # Streamlit reruns automatically after button press and state changes

    elif st.session_state.system_initialized_successfully: # If init was ATTEMPTED and SUCCEEDED
        current_llm_display = novel_core.llm_providers_map_core.get(st.session_state.selected_llm_provider_key, '未知').upper() if _CORE_IMPORTED_SUCCESSFULLY else "未知"
        current_emb_data = novel_core.embedding_providers_map_core.get(st.session_state.selected_embedding_provider_key, ["未知"]) if _CORE_IMPORTED_SUCCESSFULLY else ["未知"]
        current_emb_display = current_emb_data[0].upper()
        
        st.sidebar.success(f"系统已初始化！\nLLM: {current_llm_display}\nEmbedding: {current_emb_display}")
        if st.sidebar.button("🔄 重新初始化/切换模型", key="reinit_button_final_ui", use_container_width=True):
            st.session_state.system_initialized_attempted = False
            st.session_state.system_initialized_successfully = False
            st.session_state.milvus_initialized_core = False # Reset core-specific flags too
            st.session_state.current_chapter = 1 
            st.session_state.current_segment_number = 0
            st.session_state.current_generated_text = ""
            st.session_state.user_directive_for_current_segment_buffer = ""
            st.session_state.resume_choice_made = False 
            st.session_state.last_known_chapter = None 
            st.session_state.last_known_segment = None
            add_log_ui("用户请求重新初始化系统。")
            # Streamlit will rerun

# --- Main Writing Area ---
if st.session_state.system_initialized_attempted:
    if st.session_state.system_initialized_successfully:
        st.header("📝 小说创作区")
        # ... (Resume logic UI from previous app_ui.py - ensure it uses int for segment/chapter) ...
        
        # --- Fix for TypeError on display ---
        current_seg_num_for_display = st.session_state.get('current_segment_number', 0)
        if not isinstance(current_seg_num_for_display, int):
            add_log_ui(f"警告: current_segment_number ({type(current_seg_num_for_display).__name__}: '{current_seg_num_for_display}') 非整数. 重置为0.", "warning")
            current_seg_num_for_display = 0; st.session_state.current_segment_number = 0
        current_chap_for_display = st.session_state.get('current_chapter', 1)
        if not isinstance(current_chap_for_display, int):
            add_log_ui(f"警告: current_chapter ({type(current_chap_for_display).__name__}: '{current_chap_for_display}') 非整数. 重置为1.", "warning")
            current_chap_for_display = 1; st.session_state.current_chapter = 1
        st.info(f"当前写作进度：章节 {current_chap_for_display}, 计划生成片段号 {current_seg_num_for_display + 1}")
        
        # --- TODO: PASTE YOUR FULL MAIN WRITING AREA UI HERE ---
        # This includes:
        # - Directive input text_area
        # - "生成故事片段" button and its callback calling novel_core.core_generate_segment_text_for_ui
        # - Display area for st.session_state.current_generated_text
        # - Action buttons (采纳, 重写, 扩写, 丢弃, 下一片段指令) and their callbacks
        #   calling novel_core.core_adopt_segment_from_ui or novel_core.core_generate_with_llm (for expand)
        st.text_area("写作指令 (在此处输入)...", key="directive_main_area_placeholder", height=150)
        if st.button("生成片段 (功能待实现)", key="gen_btn_placeholder"):
            st.write("生成逻辑待连接到 novel_core.core_generate_segment_text_for_ui")

    else: # system_initialized_attempted is True, but system_initialized_successfully is False
        st.error("🤷 系统初始化失败，请检查侧边栏或下方的日志，并尝试在侧边栏重新初始化。")
elif not st.session_state.system_initialized_attempted and _CORE_IMPORTED_SUCCESSFULLY: 
    st.warning("👈 请在侧边栏选择模型并点击“初始化系统”以开始使用。")
elif not _CORE_IMPORTED_SUCCESSFULLY:
    st.error("核心模块 novel_core.py未能加载，应用无法启动。请检查控制台错误。")


# --- Log Display Area (at the bottom of the main page or in sidebar) ---
st.markdown("---")
st.subheader("运行日志 (最新在前)")
log_container_main = st.container()
log_html_content_main = "<div style='max-height: 300px; overflow-y: auto; border: 1px solid #e6e6e6; padding: 10px; font-size: 0.85em;'>"
if isinstance(st.session_state.get("log_messages"), list):
    for msg in st.session_state.log_messages: 
        color = "inherit"
        if "[ERROR]" in msg or "[FATAL]" in msg: color = "red"
        elif "[WARNING]" in msg: color = "orange"
        elif "[SUCCESS]" in msg: color = "green"
        log_html_content_main += f"<pre style='color:{color}; white-space: pre-wrap; word-wrap: break-word; margin-bottom: 2px;'>{st.html(msg)}</pre>"
else:
    log_html_content_main += "<pre style='color:red;'>错误：日志列表未正确初始化。</pre>"
log_html_content_main += "</div>"
log_container_main.markdown(log_html_content_main, unsafe_allow_html=True)

if st.button("清除主界面日志", key="clear_main_logs_button"):
    st.session_state.log_messages = ["UI主界面日志已清除。"]