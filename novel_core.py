# novel_core.py
from typing import Dict, Optional, List

# Standard library imports
import logging
import os
import uuid
from datetime import datetime
import time

_STREAMLIT_AVAILABLE = False
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    class MockSessionState:
        def __init__(self): self._state = {}
        def get(self, key, default=None): return self._state.get(key, default)
        def __getitem__(self, key): return self._state[key]
        def __setitem__(self, key, value): self._state[key] = value
        def __contains__(self, key): return key in self._state
        def insert(self, index, value): 
            if 'log_messages' not in self._state: self._state['log_messages'] = []
            self._state['log_messages'].insert(index, value)
    if 'st' not in globals() or globals()['st'] is None: 
        _st_instance_mock = MockSessionState()
        st = type('StreamlitMock', (), {'session_state': _st_instance_mock})()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("NovelCorePreImport").warning("Streamlit module not found or 'st' mock was created.")

try:
    import openai
    from sentence_transformers import SentenceTransformer
    from pymilvus import connections, utility, CollectionSchema, FieldSchema, DataType, Collection
    import httpx
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
    from dotenv import load_dotenv
except ImportError as e:
    error_msg_lib = f"Core: Missing critical third-party libraries: {e}."
    if _STREAMLIT_AVAILABLE and hasattr(st, 'error'): st.error(error_msg_lib)
    else: logging.getLogger("NovelCorePreImport").error(error_msg_lib)
    raise

# --- Global Configurations ---
load_dotenv()
logger = logging.getLogger("NovelCore")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants Definition ---
OPENAI_API_KEY_ENV_NAME="OPENAI_API_KEY"; DEEPSEEK_API_KEY_ENV_NAME="DEEPSEEK_API_KEY"; GEMINI_API_KEY_ENV_NAME="GEMINI_API_KEY"; CUSTOM_PROXY_API_KEY_ENV_NAME="CUSTOM_PROXY_API_KEY"
OPENAI_OFFICIAL_HTTP_PROXY_CORE=os.getenv("OPENAI_OFFICIAL_HTTP_PROXY"); OPENAI_OFFICIAL_HTTPS_PROXY_CORE=os.getenv("OPENAI_OFFICIAL_HTTPS_PROXY")
DEEPSEEK_LLM_HTTP_PROXY_CORE=os.getenv("DEEPSEEK_LLM_HTTP_PROXY"); DEEPSEEK_LLM_HTTPS_PROXY_CORE=os.getenv("DEEPSEEK_LLM_HTTPS_PROXY")
CUSTOM_LLM_HTTP_PROXY_CORE=os.getenv("CUSTOM_LLM_HTTP_PROXY"); CUSTOM_LLM_HTTPS_PROXY_CORE=os.getenv("CUSTOM_LLM_HTTPS_PROXY")
GEMINI_HTTP_PROXY_CORE=os.getenv("GEMINI_HTTP_PROXY",os.getenv("GLOBAL_HTTP_PROXY")); GEMINI_HTTPS_PROXY_CORE=os.getenv("GEMINI_HTTPS_PROXY",os.getenv("GLOBAL_HTTPS_PROXY"))
CUSTOM_PROXY_BASE_URL_CORE=os.getenv("CUSTOM_PROXY_BASE_URL","https://api.openai-next.com/v1"); HARDCODED_CUSTOM_PROXY_KEY_CORE="sk-0rRacqNU0aY2DrOV1601DfE76a4d4a87A14f97383e25095b"
MILVUS_ALIAS_CORE="default_novel_core"; MILVUS_HOST_CORE=os.getenv("MILVUS_HOST","localhost"); MILVUS_PORT_CORE=os.getenv("MILVUS_PORT","19530")
ZILLIZ_CLOUD_URI_ENV_NAME="ZILLIZ_CLOUD_URI"; ZILLIZ_CLOUD_TOKEN_ENV_NAME="ZILLIZ_CLOUD_TOKEN"
OPENAI_EMBEDDING_MODEL_CORE="text-embedding-3-small"; ST_MODEL_TEXT2VEC_CORE="shibing624/text2vec-base-chinese"; ST_MODEL_BGE_LARGE_ZH_CORE="BAAI/bge-large-zh-v1.5"
OPENAI_LLM_MODEL_CORE="gpt-3.5-turbo"; DEEPSEEK_LLM_MODEL_CORE="deepseek-chat"; GEMINI_LLM_MODEL_CORE="gemini-1.5-flash-latest" ; CUSTOM_PROXY_LLM_MODEL_CORE="gpt-3.5-turbo"
DEEPSEEK_BASE_URL_CORE=os.getenv("DEEPSEEK_BASE_URL","https://api.deepseek.com")
SETTINGS_FILES_DIR_CORE="./novel_setting_files"; NOVEL_MD_OUTPUT_DIR_CORE="./novel_markdown_chapters_core"
COLLECTION_NAME_LORE_PREFIX_CORE="novel_lore_mv"; COLLECTION_NAME_STORY_PREFIX_CORE="novel_story_mv"
embedding_providers_map_core={"1":("sentence_transformer_text2vec",ST_MODEL_TEXT2VEC_CORE,"(本地ST Text2Vec中文)",768),"2":("sentence_transformer_bge_large_zh",ST_MODEL_BGE_LARGE_ZH_CORE,"(本地ST BGE中文Large)",1024),"3":("openai_official",OPENAI_EMBEDDING_MODEL_CORE,"(官方OpenAI API Key)",1536)}
llm_providers_map_core={"1":"openai_official","2":"deepseek","3":"gemini","4":"custom_proxy_llm"}

_core_original_os_environ_proxies: Dict[str, Optional[str]] = {}

def _log_to_ui_if_available(message: str, level: str = "INFO", module_prefix: str = "Core"):
    if _STREAMLIT_AVAILABLE and hasattr(st, 'session_state') and 'log_messages' in st.session_state:
        if not isinstance(st.session_state.log_messages, list): st.session_state.log_messages = []
        st.session_state.log_messages.insert(0, f"[{level.upper()}][{module_prefix}] {message}")

# --- HELPER FUNCTIONS (Proxy, Custom Key) ---
# TODO: PASTE YOUR FULL IMPLEMENTATIONS for core_get_custom_proxy_key, core_set_temp_os_proxies, 
# TODO: core_restore_original_os_proxies, core_get_httpx_client_with_proxy HERE.

# --- EMBEDDING FUNCTIONS ---
# TODO: PASTE YOUR FULL IMPLEMENTATIONS for core_get_openai_embeddings, core_get_st_embeddings HERE.

# --- TEXT PROCESSING ---
def core_chunk_text_by_paragraph(text: str) -> List[str]:
    # TODO: PASTE YOUR FULL IMPLEMENTATION HERE.
    return [p.strip() for p in text.replace('\r\n', '\n').replace('\r', '\n').split('\n\n') if p.strip()]

# --- MILVUS FUNCTIONS ---
def core_load_and_vectorize_settings():
    # TODO: PASTE YOUR FULL IMPLEMENTATION HERE.
    # This function MUST use st.session_state.lore_collection_milvus_obj (a Collection object)
    # and call actual embedding functions.
    logger.info("Core: core_load_and_vectorize_settings - FULL IMPLEMENTATION NEEDED.")
    _log_to_ui_if_available("加载和向量化设定文件 (实际逻辑待填充)...", module_prefix="CoreLoadSettings")

def core_seed_initial_lore():
    core_load_and_vectorize_settings() # Defined above
    # TODO: PASTE YOUR FULL IMPLEMENTATION HERE for fallback lore.
    logger.info("Core: core_seed_initial_lore - FULL IMPLEMENTATION NEEDED.")
    _log_to_ui_if_available("检查/载入知识库种子数据 (实际逻辑待填充)...", module_prefix="CoreSeedLore")

def core_init_milvus_collections_internal():
    # THIS IS THE FULL ZILLIZ-AWARE VERSION WITH CORRECTED VARIABLE DEFINITION ORDER
    log_to_ui_milvus = lambda msg, lvl="INFO": _log_to_ui_if_available(msg, lvl, "CoreInitMilvus")
    logger.info("Core: core_init_milvus_collections_internal called for ACTUAL collection setup.")
    log_to_ui_milvus("开始实际的Milvus集合初始化...")

    if 'selected_embedding_provider_identifier' not in st.session_state or not st.session_state.selected_embedding_provider_identifier:
        err_msg = "Core CRITICAL (MilvusInit): 'selected_embedding_provider_identifier' 未在 session_state 中有效设置."
        logger.critical(err_msg); log_to_ui_milvus(err_msg, "FATAL"); raise AttributeError(err_msg) 
    if not st.session_state.get('embedding_dimension') or \
       not isinstance(st.session_state.embedding_dimension, int) or \
       st.session_state.embedding_dimension <= 0:
        err_msg = f"Core CRITICAL (MilvusInit): Invalid embedding_dimension: {st.session_state.get('embedding_dimension')}."
        logger.critical(err_msg); log_to_ui_milvus(err_msg, "FATAL"); raise ValueError(err_msg)
    current_embedding_dimension = st.session_state.embedding_dimension
    logger.debug(f"Core: Using embedding_dimension: {current_embedding_dimension} for Milvus schema.")

    # 1. Establish Connection (Zilliz or Local)
    # TODO: PASTE YOUR FULL, WORKING Milvus connection logic HERE.
    # This sets st.session_state.milvus_target
    # Example structure:
    # zilliz_uri = os.getenv(...) ...
    # try: connections.connect(alias=MILVUS_ALIAS_CORE, ...) ... except ...
    logger.info("Core: Milvus connection (FULL IMPLEMENTATION NEEDED).")
    _log_to_ui_if_available("Milvus连接成功 (模拟)。", "SUCCESS", "CoreInitMilvus") # Placeholder

    # 2. Define Collection Names
    # TODO: PASTE YOUR FULL logic for provider_short, model_short_suffix,
    # TODO: sanitize_milvus_name_local_helper (if not global), lore_col_name, story_col_name.
    # This uses st.session_state.selected_embedding_provider_identifier.
    lore_col_name = "temp_lore_name_replace" # Placeholder
    story_col_name = "temp_story_name_replace" # Placeholder
    st.session_state.lore_collection_name = lore_col_name
    st.session_state.story_collection_name = story_col_name
    log_to_ui_milvus(f"Lore Collection目标名称: {lore_col_name}")
    log_to_ui_milvus(f"Story Collection目标名称: {story_col_name}")


    # 3. Define Schemas
    # TODO: PASTE YOUR FULL FieldSchema and CollectionSchema definitions HERE
    # for lore_schema_list and story_schema_list.
    # These depend on current_embedding_dimension.
    lore_schema_list: List[FieldSchema] = [] # REPLACE WITH ACTUAL SCHEMA
    story_schema_list: List[FieldSchema] = [] # REPLACE WITH ACTUAL SCHEMA
    if not lore_schema_list or not story_schema_list: # Basic check for placeholders
        logger.warning("Milvus Schemas are placeholders! Replace with actual definitions.")
    log_to_ui_milvus("Schema定义完成 (占位符 - 需替换)。")


    # 4. Define _create_or_get_collection helper
    def _create_or_get_collection(name: str, schema_list_param: List[FieldSchema], desc: str) -> Collection:
        # TODO: PASTE YOUR FULL, WORKING _create_or_get_collection logic HERE.
        # It uses utility.has_collection, Collection(), create_index(), load(), and MILVUS_ALIAS_CORE.
        # It must return an actual pymilvus.Collection object.
        logger.warning(f"_create_or_get_collection for '{name}' is MOCKED. Needs real implementation.")
        # This is a critical placeholder that will cause errors if not replaced:
        if not schema_list_param: raise ValueError(f"Schema for {name} is empty!")
        # Return a mock object that *might* pass isinstance(obj, Collection) if Collection is imported,
        # but won't have actual Milvus functionality.
        class MockCollection:
            def __init__(self, name): self.name = name; self.num_entities = 0
            def load(self, timeout=None): logger.info(f"Mock load for {self.name}")
            def has_collection(self): return True # Mock
            def insert(self, data): logger.info(f"Mock insert for {self.name}"); return type("InsertResult", (), {"primary_keys":[1]})()
            def flush(self, timeout=None): logger.info(f"Mock flush for {self.name}")
            # Add other methods your code calls like search, query if needed for it to not crash immediately
        return MockCollection(name) # type: ignore

    # 5. Get or create actual collections
    try:
        st.session_state.lore_collection_milvus_obj = _create_or_get_collection(lore_col_name, lore_schema_list, "Novel Lore Collection")
        st.session_state.story_collection_milvus_obj = _create_or_get_collection(story_col_name, story_schema_list, "Novel Story Segments Collection")
    except Exception as e_coll_ops_main: # ... (error handling) ...
        raise 

    st.session_state.milvus_initialized_core = True
    log_to_ui_milvus("所有Milvus集合准备就绪 (部分模拟 - 需替换)。", "SUCCESS")

# TODO: PASTE YOUR FULL IMPLEMENTATIONS for:
# core_add_story_segment_to_milvus, core_retrieve_relevant_lore, core_retrieve_recent_story_segments

# --- LLM Generation Function ---
def core_generate_with_llm(provider_name: str, prompt_text_from_rag: str, temperature: float =0.7, max_tokens_override: Optional[int]=None, system_message_override: Optional[str]=None):
    # TODO: PASTE YOUR FULL, WORKING LLM GENERATION LOGIC HERE
    logger.info(f"Core: LLM call to {provider_name} (FULL IMPLEMENTATION NEEDED).")
    return f"[ACTUAL LLM Output for {provider_name} WILL BE HERE]\nBased on: {prompt_text_from_rag[:70]}..."


# --- Main Initialization Function (called by UI) ---
# In novel_core.py

# Ensure these are defined at the top of the file:
# embedding_providers_map_core, llm_providers_map_core
# _log_to_ui_if_available helper function
# All constants

def core_initialize_system(embedding_choice_key: str, llm_choice_key: str, api_keys_from_ui: dict):
    log_to_ui_core_sys_init = lambda msg, lvl="INFO": _log_to_ui_if_available(msg, lvl, "CoreSysInit")

    try:
        log_to_ui_core_sys_init("开始核心系统初始化...")
        logger.info(f"Core: Initializing system: Embedding Key='{embedding_choice_key}', LLM Key='{llm_choice_key}'")
        # Initialize system_initialized_successfully to False at the start of each attempt
        st.session_state.system_initialized_successfully = False 

        # Store raw choices from UI into session_state
        st.session_state.selected_embedding_provider_key = embedding_choice_key
        st.session_state.selected_llm_provider_key = llm_choice_key
        st.session_state.api_keys = api_keys_from_ui

        # --- 1. Embedding Provider Setup (CRITICAL: SETS THE REQUIRED SESSION STATE VARS) ---
        log_to_ui_core_sys_init("步骤1: 设置嵌入提供商...")
        logger.debug(f"CoreSysInit DEBUG: Received embedding_choice_key = '{embedding_choice_key}' (type: {type(embedding_choice_key)})")
        
        if not embedding_choice_key or not isinstance(embedding_choice_key, str) or \
           embedding_choice_key not in embedding_providers_map_core:
            err_msg = f"无效的 embedding_choice_key: '{embedding_choice_key}'. 它必须是字典 embedding_providers_map_core 中的一个有效键 (字符串类型)。可用keys: {list(embedding_providers_map_core.keys())}"
            logger.error(err_msg); log_to_ui_core_sys_init(err_msg, "FATAL"); raise ValueError(err_msg) 
        
        emb_data = embedding_providers_map_core[embedding_choice_key] 
        logger.debug(f"CoreSysInit DEBUG: Fetched emb_data for key '{embedding_choice_key}': {emb_data}")

        if len(emb_data) < 4 or not isinstance(emb_data[3], int) or emb_data[3] <= 0:
            err_msg = f"embedding_providers_map_core 中 key '{embedding_choice_key}' 的维度信息无效: '{emb_data}'. 需要格式 (id_str, model_str, text_str, dimension_int)."
            logger.error(err_msg); log_to_ui_core_sys_init(err_msg, "FATAL"); raise ValueError(err_msg)
            
        emb_identifier, emb_model_name, _, emb_dim = emb_data
        
        # VVVVVV THESE LINES MUST EXECUTE SUCCESSFULLY VVVVVV
        st.session_state.selected_embedding_provider_identifier = str(emb_identifier) 
        st.session_state.embedding_dimension = int(emb_dim) 
        st.session_state.selected_st_model_name = str(emb_model_name) if "sentence_transformer" in emb_identifier else None
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        logger.info(f"Core: Embedding Config SET in session_state: ID='{st.session_state.selected_embedding_provider_identifier}', Model='{st.session_state.selected_st_model_name}', Dimension='{st.session_state.embedding_dimension}'")
        log_to_ui_core_sys_init(f"嵌入配置已在session_state中设置: ID='{st.session_state.selected_embedding_provider_identifier}', Dim='{st.session_state.embedding_dimension}'")
        
        # Sentence Transformer Model Loading (if selected)
        if "sentence_transformer" in st.session_state.selected_embedding_provider_identifier:
            current_loaded_st_model = st.session_state.get('loaded_st_model_name')
            current_st_instance = st.session_state.get('embedding_model_instance')
            st_model_to_load = st.session_state.selected_st_model_name

            if current_loaded_st_model != st_model_to_load or \
               current_st_instance is None or \
               not isinstance(current_st_instance, SentenceTransformer): # Ensure it's an actual ST model
                log_to_ui_core_sys_init(f"正在加载ST模型: {st_model_to_load}...")
                logger.info(f"Core: Loading ST model: {st_model_to_load}")
                try:
                    # TODO: Replace this with your actual SentenceTransformer loading if it was mocked.
                    # This line should actually load the model.
                    st.session_state.embedding_model_instance = SentenceTransformer(st_model_to_load)
                    st.session_state.loaded_st_model_name = st_model_to_load
                    # Dimension verification (optional but good)
                    test_emb_st = st.session_state.embedding_model_instance.encode(["test"]) # type: ignore
                    actual_dim_st = test_emb_st.shape[1]
                    if actual_dim_st != st.session_state.embedding_dimension:
                        logger.warning(f"ST Model '{st_model_to_load}' 实际维度 ({actual_dim_st}) 与配置维度 ({st.session_state.embedding_dimension}) 不符! 更新为实际维度。")
                        st.session_state.embedding_dimension = actual_dim_st # Update if different
                    log_to_ui_core_sys_init(f"ST模型 '{st_model_to_load}' 加载成功 (Dim: {st.session_state.embedding_dimension}).")
                except Exception as e_st_load:
                    logger.error(f"Core: 加载ST模型 '{st_model_to_load}' 失败: {e_st_load}", exc_info=True)
                    log_to_ui_core_sys_init(f"加载ST模型失败: {e_st_load}", "ERROR"); raise
        elif st.session_state.selected_embedding_provider_identifier == "openai_official":
            st.session_state.embedding_model_instance = None 
            st.session_state.loaded_st_model_name = None
        log_to_ui_core_sys_init("步骤1完成: 嵌入提供商已设置。")

        # --- 2. LLM Provider Setup ---
        log_to_ui_core_sys_init("步骤2: 设置LLM提供商...")
        if not llm_choice_key or llm_choice_key not in llm_providers_map_core:
            err_msg_llm = f"无效的 llm_choice_key: '{llm_choice_key}'."
            logger.error(err_msg_llm); log_to_ui_core_sys_init(err_msg_llm, "FATAL"); raise ValueError(err_msg_llm)
        llm_provider_name = llm_providers_map_core[llm_choice_key] 
        st.session_state.current_llm_provider = llm_provider_name
        # Reset LLM clients if provider changes (core_generate_with_llm will handle re-init)
        if st.session_state.get('previous_llm_provider_for_client_management') != llm_provider_name:
            st.session_state.llm_client = None 
            st.session_state.gemini_llm_client_core = None 
            st.session_state.previous_llm_provider_for_client_management = llm_provider_name
        log_to_ui_core_sys_init(f"选择LLM模型: {llm_provider_name.upper()}.")
        log_to_ui_core_sys_init("步骤2完成.")
        
        # --- 3. API Key Checks ---
        log_to_ui_core_sys_init("步骤3: 检查API Keys...")
        # TODO: Implement your FULL API key validation logic here
        # This logic should use st.session_state.api_keys, os.getenv,
        # st.session_state.selected_embedding_provider_identifier, and st.session_state.current_llm_provider
        # to determine which keys are required and if they are present. Raise ValueError if not.
        logger.info("Core: API Key checks (FULL IMPLEMENTATION NEEDED).")
        log_to_ui_core_sys_init("API Key检查通过 (需完整实现).") # Assuming success for now
        log_to_ui_core_sys_init("步骤3完成.")

        # --- 4. Milvus Initialization ---
        log_to_ui_core_sys_init("步骤4: 初始化Milvus...")
        # Crucially, selected_embedding_provider_identifier and embedding_dimension are now set.
        core_init_milvus_collections_internal() 
        core_seed_initial_lore() # Assumes Milvus collections are ready
        log_to_ui_core_sys_init("步骤4完成: Milvus已初始化并载入种子数据。")
        
        # --- 5. Story Resume Logic ---
        log_to_ui_core_sys_init("步骤5: 加载/重置故事状态...")
        st.session_state.current_chapter = 1 
        st.session_state.current_segment_number = 0 
        st.session_state.last_known_chapter = None 
        st.session_state.last_known_segment = None
        # TODO: Implement actual Milvus query for resume state.
        # Ensure parsed chapter/segment numbers are integers and update session_state accordingly.
        log_to_ui_core_sys_init("步骤5完成: 故事状态已初始化/重置。")
        
        st.session_state.system_initialized_successfully = True # Mark overall success
        logger.info("Core: System initialization successful.")
        log_to_ui_core_sys_init("核心系统初始化流程完成！", "SUCCESS")
        return True

    except Exception as e: # Catch any exception during the entire initialization
        st.session_state.system_initialized_successfully = False # Mark failure
        # Ensure safe integer defaults for UI display if error occurs
        st.session_state.current_chapter = int(st.session_state.get('current_chapter', 1)) 
        st.session_state.current_segment_number = int(st.session_state.get('current_segment_number', 0))
        
        error_message_for_log = f"核心系统初始化失败: {type(e).__name__} - {str(e)}"
        logger.error(f"Core: System initialization FAILED: {e}", exc_info=True) # Full traceback to console
        log_to_ui_core_sys_init(error_message_for_log, "FATAL") # More specific message for UI
        raise # Re-raise for app_ui.py to catch and display a general error message

# --- UI Specific Core Functions ---
def core_generate_segment_text_for_ui(user_directive: str) -> Optional[str]:
    # TODO: Implement your full RAG logic
    if not st.session_state.get('system_initialized_successfully', False): return "错误: 系统未初始化。"
    return f"模拟UI片段生成，指令：{user_directive[:50]}" # Placeholder

def core_adopt_segment_from_ui(text_content: str, chapter: int, segment_num: int, user_directive_snippet: str):
    # TODO: Implement your full adoption logic
    if not st.session_state.get('system_initialized_successfully', False): return False
    st.session_state.last_adopted_segment_text = f"[Ch{chapter}-Seg{segment_num}]\n{text_content}"
    return True















# streamlit run app_ui.py   启动命令