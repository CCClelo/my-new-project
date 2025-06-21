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
    page_icon="✍️", # You can use an emoji or a URL to an image
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help', # Replace with your help URL
        'Report a bug': "https://www.example.com/bug", # Replace
        'About': "# AI小说写作助手\n这是一个AI辅助写作工具。"
    }
)

# --- 2. Attempt to Import Core Logic (novel_core.py) ---
_CORE_IMPORTED_SUCCESSFULLY = False
_CORE_IMPORT_ERROR_MESSAGE = ""
_CORE_IMPORT_ERROR_TRACEBACK = ""
try:
    # Configure basic logging for app_ui BEFORE importing novel_core
    # so novel_core can also use this logger if it wants (though it has its own)
    ui_logger = logging.getLogger("AppUI") 
    if not ui_logger.hasHandlers():
        logging.basicConfig(level=logging.DEBUG, # Or INFO
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ui_logger.info("app.py: Script execution started. Attempting to import novel_core...")
    
    import novel_core # This is the critical import
    _CORE_IMPORTED_SUCCESSFULLY = True
    ui_logger.info("app.py: novel_core imported successfully!")

except ImportError as e_imp:
    _CORE_IMPORT_ERROR_MESSAGE = f"CRITICAL ERROR: 无法导入 novel_core.py。请确保该文件与 app.py 在同一目录下，并且所有依赖（如openai, pymilvus等）已在requirements.txt中声明。错误: {e_imp}"
    _CORE_IMPORT_ERROR_TRACEBACK = traceback.format_exc()
    # Log to console as st.error might not be usable yet if Streamlit itself has issues
    if 'ui_logger' in globals(): ui_logger.critical(_CORE_IMPORT_ERROR_MESSAGE, exc_info=True)
    else: print(_CORE_IMPORT_ERROR_MESSAGE + "\n" + _CORE_IMPORT_ERROR_TRACEBACK)
except Exception as e_load: 
    _CORE_IMPORT_ERROR_MESSAGE = f"CRITICAL ERROR: 导入或执行 novel_core.py 顶层代码时发生意外错误: {e_load}"
    _CORE_IMPORT_ERROR_TRACEBACK = traceback.format_exc()
    if 'ui_logger' in globals(): ui_logger.critical(_CORE_IMPORT_ERROR_MESSAGE, exc_info=True)
    else: print(_CORE_IMPORT_ERROR_MESSAGE + "\n" + _CORE_IMPORT_ERROR_TRACEBACK)

# --- 3. Define Placeholders if Core Logic Failed to Import ---
if not _CORE_IMPORTED_SUCCESSFULLY:
    class NovelCorePlaceholder: # Used if novel_core.py itself has fatal import-time errors
        embedding_providers_map_core = {"0": ("错误", "错误", "核心模块加载失败", 0)}
        llm_providers_map_core = {"0": "错误"}
        NOVEL_MD_OUTPUT_DIR_CORE = "./novel_markdown_placeholder"
        # Add stubs for functions that initialize_session_state_ui might call if needed
        def core_initialize_system(self, *args, **kwargs): raise RuntimeError("novel_core未加载，无法初始化。")
        def core_generate_segment_text_for_ui(self, *args, **kwargs): return "novel_core未加载。"
        def core_adopt_segment_from_ui(self, *args, **kwargs): return False
    novel_core = NovelCorePlaceholder()


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
        "log_messages": ["UI应用已启动，等待初始化..."], # This will be the first message
        "current_chapter": 1, "current_segment_number": 0, 
        "current_generated_text": "", "user_directive_for_current_segment": "",
        "selected_embedding_provider_key": default_emb_key,
        "selected_llm_provider_key": default_llm_key,
        "api_keys": {}, 
        "max_tokens_per_llm_call": int(os.getenv("MAX_TOKENS_PER_LLM_CALL", "7800")),
        "segments_per_chapter_advance": int(os.getenv("SEGMENTS_PER_CHAPTER_ADVANCE", "3")),
        "last_adopted_segment_text": "这是故事的开端，尚无先前的故事片段。",
        "user_directive_for_current_segment_buffer": "", "show_expand_input": False, 
        "num_recent_segments_to_fetch_ui": 2, "llm_temperature": 0.7, 
        "novel_md_output_dir_ui": md_dir,
        "last_known_chapter": None, "last_known_segment": None, 
        "resume_choice_made": False, "resume_choice_idx": 0 
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

initialize_session_state_ui() # Call it once at the start of the UI script

# --- Helper function to add logs to UI and console logger ---
def add_log_ui(message: str, level: str = "info", UImodule: str = "UI"):
    console_logger = logging.getLogger("AppUI") 
    timestamp = f"[{UImodule}][{st.session_state.get('current_llm_provider', 'SYS') if _CORE_IMPORTED_SUCCESSFULLY and st.session_state.get('system_initialized_successfully') else 'UI_INIT'}] "
    log_entry = f"{timestamp}{message}"
    if not isinstance(st.session_state.get("log_messages"), list): st.session_state.log_messages = ["Log list init error in add_log_ui."]
    st.session_state.log_messages.insert(0, log_entry)
    
    if level.lower() == "info": console_logger.info(message)
    elif level.lower() == "warning": console_logger.warning(message)
    elif level.lower() == "error": console_logger.error(message)
    elif level.lower() == "fatal": console_logger.critical(message) 
    elif level.lower() == "debug": console_logger.debug(message)
    elif level.lower() == "success": console_logger.info(f"SUCCESS: {message}")


# --- 5. Display Critical Import Error if Occurred ---
if not _CORE_IMPORTED_SUCCESSFULLY:
    st.title("AI 小说写作助手 - 启动失败")
    st.error(_CORE_IMPORT_ERROR_MESSAGE)
    if _CORE_IMPORT_ERROR_TRACEBACK:
        st.subheader("错误详情 (Traceback):")
        st.code(_CORE_IMPORT_ERROR_TRACEBACK)
    st.warning("请检查应用的日志（在Hugging Face Spaces的Logs标签页，或本地运行时的控制台）获取更多信息。确保所有依赖已在 `requirements.txt` 中正确声明并安装成功。")
    st.stop() # Stop further UI rendering if core module is missing

# --- UI Layout (Assuming _CORE_IMPORTED_SUCCESSFULLY is True from here) ---
st.title("✍️ AI 小说写作助手")

with st.sidebar:
    # ... (Full sidebar implementation from your previous complete app_ui.py)
    # This includes: Embedding Selector, LLM Selector, Advanced Config (API Keys, tokens, temp),
    # Init Button, Re-Init Button, Log Display.
    # Ensure all st.session_state keys used here are defined in initialize_session_state_ui.
    # Example snippet for init button:
    st.header("系统配置")
    # ... (Selectors for embedding and LLM, storing choices in st.session_state.selected_..._key) ...
    if not st.session_state.system_initialized_attempted:
        if st.sidebar.button("🚀 初始化系统", key="init_button_final_v3_ui", type="primary", use_container_width=True):
            st.session_state.system_initialized_attempted = True
            with st.spinner("系统初始化中... (可能需要较长时间)"):
                try:
                    add_log_ui(f"UI 发起初始化。选择的嵌入Key: '{st.session_state.selected_embedding_provider_key}', LLM Key: '{st.session_state.selected_llm_provider_key}'")
                    novel_core.core_initialize_system(
                        st.session_state.selected_embedding_provider_key,
                        st.session_state.selected_llm_provider_key,
                        st.session_state.api_keys 
                    )
                except Exception as e_init_call_ui: 
                    st.session_state.system_initialized_successfully = False
                    add_log_ui(f"UI层面捕获到初始化失败: {e_init_call_ui}", "error")
    # ... (rest of sidebar logic)

# --- Main Writing Area ---
if st.session_state.system_initialized_attempted:
    if st.session_state.system_initialized_successfully:
        st.header("📝 小说创作区")
        # ... (Full resume logic UI from your previous complete app_ui.py) ...
        # ... (Full TypeError fix for current_segment_number display from previous) ...
        # ... (Full Directive input, Generate button, Display Area, Action buttons -
        #      PASTE YOUR FULL IMPLEMENTATION for these UI elements and their callbacks here.
        #      Ensure callbacks call the correct novel_core.core_..._for_ui functions)
        st.info("小说创作区 (实际UI和功能待填充，但核心系统已初始化)。") # Placeholder
    else: 
        st.error("🤷 系统初始化失败，请检查侧边栏或下方的日志，并尝试在侧边栏重新初始化。")
elif not st.session_state.system_initialized_attempted: 
    st.warning("👈 请在侧边栏选择模型并点击“初始化系统”以开始使用。")

# --- Log Display Area (at the bottom of the main page) ---
st.markdown("---")
st.subheader("运行日志 (最新在前)")
# ... (Full log display HTML/Markdown code from your previous complete app_ui.py) ...
# Example:
log_container_main = st.container()
# ... (rest of log display logic) ...

add_log_ui("app.py: 脚本渲染到达末尾。", "debug")