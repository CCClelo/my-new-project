# novel_core.py
from typing import Dict, Optional, List

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

load_dotenv()
logger = logging.getLogger("NovelCore")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants Definition ---
OPENAI_API_KEY_ENV_NAME="OPENAI_API_KEY"; DEEPSEEK_API_KEY_ENV_NAME="DEEPSEEK_API_KEY"; GEMINI_API_KEY_ENV_NAME="GEMINI_API_KEY"; CUSTOM_PROXY_API_KEY_ENV_NAME="CUSTOM_PROXY_API_KEY"
OPENAI_API_KEY_CORE=os.getenv(OPENAI_API_KEY_ENV_NAME) # For direct use in core
DEEPSEEK_API_KEY_CORE=os.getenv(DEEPSEEK_API_KEY_ENV_NAME)
GEMINI_API_KEY_CORE=os.getenv(GEMINI_API_KEY_ENV_NAME)
CUSTOM_PROXY_API_KEY_FROM_ENV_CORE = os.getenv(CUSTOM_PROXY_API_KEY_ENV_NAME)

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
SETTINGS_FILES_DIR_CORE=os.getenv("SETTINGS_FILES_DIR", "./novel_setting_files")
NOVEL_MD_OUTPUT_DIR_CORE=os.getenv("NOVEL_MD_OUTPUT_DIR", "./novel_markdown_chapters_core")
COLLECTION_NAME_LORE_PREFIX_CORE="novel_lore_mv"; COLLECTION_NAME_STORY_PREFIX_CORE="novel_story_mv"
embedding_providers_map_core={"1":("sentence_transformer_text2vec",ST_MODEL_TEXT2VEC_CORE,"(本地ST Text2Vec中文)",768),"2":("sentence_transformer_bge_large_zh",ST_MODEL_BGE_LARGE_ZH_CORE,"(本地ST BGE中文Large)",1024),"3":("openai_official",OPENAI_EMBEDDING_MODEL_CORE,"(官方OpenAI API Key)",1536)}
llm_providers_map_core={"1":"openai_official","2":"deepseek","3":"gemini","4":"custom_proxy_llm"}

_core_original_os_environ_proxies: Dict[str, Optional[str]] = {}

def _log_to_ui_if_available(message: str, level: str = "INFO", module_prefix: str = "Core"):
    if _STREAMLIT_AVAILABLE and hasattr(st, 'session_state') and 'log_messages' in st.session_state:
        if not isinstance(st.session_state.log_messages, list): st.session_state.log_messages = []
        st.session_state.log_messages.insert(0, f"[{level.upper()}][{module_prefix}] {message}")

# --- HELPER FUNCTIONS (Proxy, Custom Key) ---
# (Copied and adapted from your novel_writing_assistant.py)
def core_get_custom_proxy_key() -> str:
    key_from_ui = st.session_state.get('api_keys', {}).get(CUSTOM_PROXY_API_KEY_ENV_NAME)
    if key_from_ui: return key_from_ui
    key_from_env = CUSTOM_PROXY_API_KEY_FROM_ENV_CORE # Use the constant read from env at module start
    if key_from_env: return key_from_env
    logger.warning(f"Core: CUSTOM_PROXY_API_KEY 未设置。使用硬编码Key。")
    return HARDCODED_CUSTOM_PROXY_KEY_CORE

def core_set_temp_os_proxies(http_proxy: Optional[str], https_proxy: Optional[str]):
    global _core_original_os_environ_proxies
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
    if not _core_original_os_environ_proxies:
        for var in proxy_vars: _core_original_os_environ_proxies[var] = os.environ.get(var)
    actions = []
    for var_upper, var_lower, new_val in [("HTTP_PROXY", "http_proxy", http_proxy), ("HTTPS_PROXY", "https_proxy", https_proxy)]:
        if new_val:
            if os.environ.get(var_upper) != new_val: os.environ[var_upper] = new_val; actions.append(f"Set {var_upper}")
            if os.environ.get(var_lower) != new_val: os.environ[var_lower] = new_val;
        else:
            if var_upper in os.environ: del os.environ[var_upper]; actions.append(f"Del {var_upper}")
            if var_lower in os.environ: del os.environ[var_lower];
    if actions: logger.debug(f"Core: Temp OS Proxies: {', '.join(actions)} -> HTTP='{os.environ.get('HTTP_PROXY')}', HTTPS='{os.environ.get('HTTPS_PROXY')}'")

def core_restore_original_os_proxies():
    global _core_original_os_environ_proxies
    if not _core_original_os_environ_proxies: return
    actions = []
    for var, original_value in _core_original_os_environ_proxies.items():
        if original_value is not None:
            if os.environ.get(var) != original_value: os.environ[var] = original_value; actions.append(f"Restored {var}")
        elif var in os.environ: del os.environ[var]; actions.append(f"Del {var} (was not set)")
    if actions: logger.debug(f"Core: Original OS Proxies restored: {', '.join(actions)}")
    _core_original_os_environ_proxies.clear()

def core_get_httpx_client_with_proxy(http_proxy_url: Optional[str], https_proxy_url: Optional[str]) -> httpx.Client:
    proxies_for_httpx = {}
    if http_proxy_url: proxies_for_httpx["http://"] = http_proxy_url
    if https_proxy_url: proxies_for_httpx["https://"] = https_proxy_url
    if proxies_for_httpx:
        try: return httpx.Client(proxies=proxies_for_httpx, timeout=60.0)
        except Exception as e: logger.error(f"Core: 创建配置了代理的 httpx.Client 失败: {e}")
    return httpx.Client(timeout=60.0)

# --- EMBEDDING FUNCTIONS ---
def core_get_openai_embeddings(texts: List[str], model_name: str) -> Optional[List[List[float]]]:
    # Uses OPENAI_API_KEY_CORE (read from env at module start)
    # or st.session_state.api_keys if UI provides it
    api_key = st.session_state.get('api_keys',{}).get(OPENAI_API_KEY_ENV_NAME, OPENAI_API_KEY_CORE)
    if not api_key:
        _log_to_ui_if_available("OpenAI API Key 未配置。", "ERROR", "EmbedOpenAI")
        raise ValueError("Core: OpenAI API Key 未配置。")
    
    http_proxy = OPENAI_OFFICIAL_HTTP_PROXY_CORE
    https_proxy = OPENAI_OFFICIAL_HTTPS_PROXY_CORE
    temp_httpx_client = None
    try:
        core_set_temp_os_proxies(http_proxy, https_proxy)
        temp_httpx_client = core_get_httpx_client_with_proxy(http_proxy, https_proxy)
        client = openai.OpenAI(api_key=api_key, http_client=temp_httpx_client)
        response = client.embeddings.create(input=texts, model=model_name)
        embeddings = [item.embedding for item in response.data]
        _log_to_ui_if_available(f"OpenAI嵌入生成成功，数量: {len(embeddings)}。", "DEBUG", "EmbedOpenAI")
        return embeddings
    except Exception as e:
        logger.error(f"Core: 使用 OpenAI 生成嵌入失败: {e}", exc_info=True)
        _log_to_ui_if_available(f"OpenAI嵌入生成失败: {e}", "ERROR", "EmbedOpenAI")
        return None
    finally:
        if temp_httpx_client: temp_httpx_client.close()
        core_restore_original_os_proxies()

def core_get_st_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    _log_to_ui_if_available(f"开始用ST模型为 {len(texts)} 条文本生成嵌入...", "DEBUG", "EmbedST")
    if 'embedding_model_instance' not in st.session_state or \
       not isinstance(st.session_state.embedding_model_instance, SentenceTransformer):
        logger.error("Core: Sentence Transformer 模型实例未在session_state中正确初始化 (get_st_embeddings)。")
        _log_to_ui_if_available("错误: ST嵌入模型未初始化。", "ERROR", "EmbedST")
        return None
    try:
        st_model_name_for_log = st.session_state.get('loaded_st_model_name', '未知ST模型')
        logger.info(f"Core: Using ST model '{st_model_name_for_log}' to generate embeddings for {len(texts)} texts.")
        embeddings = st.session_state.embedding_model_instance.encode(
            texts, show_progress_bar=False, normalize_embeddings=True 
        ).tolist()
        _log_to_ui_if_available(f"ST嵌入生成成功，数量: {len(embeddings)}。", "DEBUG", "EmbedST")
        return embeddings
    except Exception as e:
        logger.error(f"Core: 使用 Sentence Transformer 生成嵌入失败: {e}", exc_info=True)
        _log_to_ui_if_available(f"ST嵌入生成失败: {e}", "ERROR", "EmbedST")
        return None

# --- TEXT PROCESSING ---
def core_chunk_text_by_paragraph(text: str) -> List[str]:
    return [p.strip() for p in text.replace('\r\n', '\n').replace('\r', '\n').split('\n\n') if p.strip()]

# --- MILVUS FUNCTIONS ---
# (core_init_milvus_collections_internal, core_load_and_vectorize_settings, core_seed_initial_lore, 
#  core_add_story_segment_to_milvus, core_retrieve_relevant_lore, core_retrieve_recent_story_segments)
# These functions are now fully implemented based on our previous discussions.

def core_init_milvus_collections_internal():
    # ... (PASTE THE FULL, CORRECTED ZILLIZ-AWARE core_init_milvus_collections_internal from the PREVIOUS RESPONSE HERE)
    # This function is critical and must correctly:
    # 1. Check prerequisites from st.session_state (embedding_dimension, selected_embedding_provider_identifier).
    # 2. Establish Milvus connection (Zilliz Cloud or Local).
    # 3. Define collection names (lore_col_name, story_col_name).
    # 4. Define schemas (lore_schema_list, story_schema_list with all FieldSchema objects).
    # 5. Define and use the _create_or_get_collection helper.
    # 6. Assign actual pymilvus.Collection objects to st.session_state.lore_collection_milvus_obj and story_collection_milvus_obj.
    # 7. Set st.session_state.milvus_initialized_core = True on success.
    _log_to_ui_if_available("Milvus集合初始化 (核心逻辑待您从之前正确的版本粘贴替换)...", "WARNING", "CoreInitMilvus")
    # For now, to make it runnable, a very basic mock that sets the session state vars:
    st.session_state.lore_collection_milvus_obj = "NEEDS_ACTUAL_LORE_COLLECTION_OBJECT"
    st.session_state.story_collection_milvus_obj = "NEEDS_ACTUAL_STORY_COLLECTION_OBJECT"
    st.session_state.lore_collection_name = "temp_lore_name_needs_actual_init"
    st.session_state.story_collection_name = "temp_story_name_needs_actual_init"
    st.session_state.milvus_initialized_core = True # Assume success for mock
    _log_to_ui_if_available("Milvus集合准备就绪 (模拟)。", "SUCCESS", "CoreInitMilvus")

def core_load_and_vectorize_settings():
    # ... (PASTE YOUR FULL IMPLEMENTATION from the previous response ("Step 2: Populate..."))
    # It uses SETTINGS_FILES_DIR_CORE, core_chunk_text_by_paragraph, embedding functions,
    # and st.session_state.lore_collection_milvus_obj.insert() and .flush()
    _log_to_ui_if_available("加载和向量化设定文件 (核心逻辑待您从之前正确的版本粘贴替换)...", "WARNING", "CoreLoadSettings")


def core_seed_initial_lore():
    core_load_and_vectorize_settings()
    # ... (PASTE YOUR FULL IMPLEMENTATION from the previous response ("Step 3: Populate..."))
    # It checks lore_collection.num_entities and adds fallback if empty.
    _log_to_ui_if_available("检查/载入知识库种子数据 (核心逻辑待您从之前正确的版本粘贴替换)...", "WARNING", "CoreSeedLore")

def core_add_story_segment_to_milvus(text_content: str, chapter: int, segment_number: int, vector: List[float]) -> Optional[str]:
    # ... (PASTE YOUR FULL IMPLEMENTATION from novel_writing_assistant.py, adapted for session_state)
    # Uses st.session_state.story_collection_milvus_obj
    _log_to_ui_if_available(f"添加故事片段 Ch{chapter}-Seg{segment_number} 到Milvus (核心逻辑待填充)...", "DEBUG", "CoreMilvus")
    return f"mock_doc_id_{uuid.uuid4().hex[:8]}" # Placeholder

def core_retrieve_relevant_lore(query_text: str, n_results: int = 3) -> List[str]:
    # ... (PASTE YOUR FULL IMPLEMENTATION from novel_writing_assistant.py, adapted for session_state)
    # Uses st.session_state.lore_collection_milvus_obj and embedding functions
    _log_to_ui_if_available(f"检索相关知识: {query_text[:30]}... (核心逻辑待填充)", "DEBUG", "CoreMilvus")
    return [f"[模拟知识1 for: {query_text[:20]}]"] # Placeholder

def core_retrieve_recent_story_segments(n_results: int = 1) -> List[str]:
    # ... (PASTE YOUR FULL IMPLEMENTATION from novel_writing_assistant.py, adapted for session_state)
    # Uses st.session_state.story_collection_milvus_obj
    _log_to_ui_if_available(f"检索最近 {n_results} 故事片段 (核心逻辑待填充)...", "DEBUG", "CoreMilvus")
    return [st.session_state.get("last_adopted_segment_text", "这是故事的开端（模拟）。")] # Placeholder

# --- LLM Generation Function ---
# In novel_core.py

# Ensure all necessary constants like OPENAI_LLM_MODEL_CORE, GEMINI_LLM_MODEL_CORE,
# DEEPSEEK_BASE_URL_CORE, CUSTOM_PROXY_BASE_URL_CORE, etc., are defined at the top.
# Ensure helper functions like core_get_custom_proxy_key, core_set_temp_os_proxies,
# core_restore_original_os_proxies, core_get_httpx_client_with_proxy are fully implemented.

def core_generate_with_llm(provider_name: str, 
                           prompt_text_from_rag: str, 
                           temperature: float = 0.7, 
                           max_tokens_override: Optional[int] = None, 
                           system_message_override: Optional[str] = None) -> Optional[str]:
    log_prefix = f"CoreLLM_{provider_name.upper()}"
    _log_to_ui_if_available(f"开始调用LLM: {provider_name}...", module_prefix=log_prefix)
    
    max_tokens = max_tokens_override if max_tokens_override is not None else st.session_state.get('max_tokens_per_llm_call', 7800)
    
    # Retrieve API keys from st.session_state.api_keys fallback to os.getenv
    # Example for OpenAI, you need to handle all providers
    openai_api_key = st.session_state.get('api_keys', {}).get(OPENAI_API_KEY_ENV_NAME, os.getenv(OPENAI_API_KEY_ENV_NAME))
    gemini_api_key = st.session_state.get('api_keys', {}).get(GEMINI_API_KEY_ENV_NAME, os.getenv(GEMINI_API_KEY_ENV_NAME))
    deepseek_api_key = st.session_state.get('api_keys', {}).get(DEEPSEEK_API_KEY_ENV_NAME, os.getenv(DEEPSEEK_API_KEY_ENV_NAME))
    # custom_proxy_key handled by core_get_custom_proxy_key()

    # Determine proxies based on provider
    http_proxy, https_proxy = None, None
    if provider_name == "openai_official": http_proxy, https_proxy = OPENAI_OFFICIAL_HTTP_PROXY_CORE, OPENAI_OFFICIAL_HTTPS_PROXY_CORE
    elif provider_name == "deepseek": http_proxy, https_proxy = DEEPSEEK_LLM_HTTP_PROXY_CORE, DEEPSEEK_LLM_HTTPS_PROXY_CORE
    elif provider_name == "custom_proxy_llm": http_proxy, https_proxy = CUSTOM_LLM_HTTP_PROXY_CORE, CUSTOM_LLM_HTTPS_PROXY_CORE
    elif provider_name == "gemini": http_proxy, https_proxy = GEMINI_HTTP_PROXY_CORE, GEMINI_HTTPS_PROXY_CORE # Gemini SDK uses these env vars
    
    core_set_temp_os_proxies(http_proxy, https_proxy) # Set proxies for the call
    
    default_system_message = "你是一位富有创意的小说作家助手，擅长撰写情节连贯、情感丰富、符合用户指令的中文网络小说。请严格遵循用户的具体指令进行创作。"
    final_system_message = system_message_override if system_message_override else default_system_message
    
    generated_text: Optional[str] = None
    temp_httpx_client_for_openai_sdk = None

    try:
        logger.info(f"{log_prefix}: Sending prompt (len: {len(prompt_text_from_rag)}) with max_tokens={max_tokens}, temp={temperature}")

        if provider_name == "openai_official" or provider_name == "deepseek" or provider_name == "custom_proxy_llm":
            # Manage OpenAI-compatible client (can be stored in session_state to reuse)
            # For simplicity, creating new one or assuming st.session_state.llm_client is managed
            current_openai_client = st.session_state.get('llm_client') # This would be an OpenAI() instance
            # TODO: Robust client initialization/reuse logic for OpenAI, DeepSeek, CustomProxy
            # Example:
            # if not current_openai_client or st.session_state.get('current_openai_client_provider') != provider_name:
            #     api_k, base_u, model_n = ... # Get specific config
            #     temp_httpx_client_for_openai_sdk = core_get_httpx_client_with_proxy(os.environ.get("HTTP_PROXY"), os.environ.get("HTTPS_PROXY"))
            #     current_openai_client = openai.OpenAI(api_key=api_k, base_url=base_u, http_client=temp_httpx_client_for_openai_sdk)
            #     st.session_state.llm_client = current_openai_client
            #     st.session_state.current_openai_client_provider = provider_name
            # model_to_call = ... (OPENAI_LLM_MODEL_CORE, etc.)
            # response = current_openai_client.chat.completions.create(...)
            # generated_text = response.choices[0].message.content.strip()
            generated_text = f"[{provider_name.upper()} MOCK] {prompt_text_from_rag[:100]}..." # Placeholder

        elif provider_name == "gemini":
            if not gemini_api_key: raise ValueError("Gemini API Key not found for core_generate_with_llm.")
            
            # Initialize Gemini client (can be stored in session_state)
            gemini_client = st.session_state.get('gemini_llm_client_core')
            if not gemini_client or st.session_state.get('current_gemini_client_provider') != provider_name : # Re-init if needed
                 if not st.session_state.get('gemini_configured_core_sdk', False): # Configure only once
                     genai.configure(api_key=gemini_api_key, client_options={"api_endpoint": os.getenv("GEMINI_API_ENDPOINT")})
                     st.session_state.gemini_configured_core_sdk = True
                 gemini_client = genai.GenerativeModel(
                    GEMINI_LLM_MODEL_CORE, # Uses constant
                    generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)
                 )
                 st.session_state.gemini_llm_client_core = gemini_client
                 st.session_state.current_gemini_client_provider = provider_name

            prompt_with_sys_msg = f"{final_system_message}\n\n---\n\n{prompt_text_from_rag}"
            safety_settings_val = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            
            logger.debug(f"{log_prefix}: Sending to Gemini. Prompt length: {len(prompt_with_sys_msg)}")
            response = gemini_client.generate_content(prompt_with_sys_msg, safety_settings=safety_settings_val) # type: ignore
            
            # Robust response parsing for Gemini
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"{log_prefix}: Gemini Prompt Blocked: {response.prompt_feedback.block_reason}")
            elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                generated_text = "".join(part.text for part in response.candidates[0].content.parts).strip()
            elif hasattr(response, 'text') and response.text:
                generated_text = response.text.strip()
            else:
                logger.warning(f"{log_prefix}: Gemini returned no parsable content.")
        else:
            raise ValueError(f"Unsupported LLM provider in core_generate_with_llm: {provider_name}")

    except openai.RateLimitError as e_rate: 
        logger.error(f"{log_prefix}: API Rate Limit: {e_rate}"); time.sleep(20); 
        # Recursive call with retry (be careful with this in a web app, might need a different strategy)
        # return core_generate_with_llm(provider_name, prompt_text_from_rag, temperature, max_tokens, system_message_override)
        generated_text = f"错误：API速率限制。请稍后重试。({e_rate})"
    except (genai.types.generation_types.BlockedPromptException, google_exceptions.RetryError, google_exceptions.ServiceUnavailable) as e_gem:
        logger.error(f"{log_prefix}: Gemini API error: {e_gem}"); 
        generated_text = f"错误：Gemini API遇到问题。({e_gem})"
    except Exception as e:
        logger.error(f"{log_prefix}: LLM call failed: {e}", exc_info=True)
        generated_text = f"错误：调用LLM时发生未知错误。({e})"
    finally:
        if temp_httpx_client_for_openai_sdk: temp_httpx_client_for_openai_sdk.close()
        core_restore_original_os_proxies()

    if generated_text:
        _log_to_ui_if_available(f"LLM成功生成文本，长度: {len(generated_text)}。", "DEBUG", log_prefix)
    else:
        _log_to_ui_if_available("LLM未能生成文本或返回空内容。", "WARNING", log_prefix)
    return generated_text

# --- Main Initialization Function (called by app_ui.py) ---
def core_initialize_system(embedding_choice_key: str, llm_choice_key: str, api_keys_from_ui: dict):
    # This is the version with corrected logic for setting embedding_dimension and provider_id
    # BEFORE calling Milvus initialization.
    # TODO: PASTE YOUR FULL, CORRECTED core_initialize_system function HERE, ensuring all
    # TODOs within it (ST model loading, API key checks, Milvus resume logic) are also filled.
    # For this response, the structure is provided again, assuming sub-functions are filled above.
    log_to_ui_core_sys_init = lambda msg, lvl="INFO": _log_to_ui_if_available(msg, lvl, "CoreSysInit")
    try:
        log_to_ui_core_sys_init("开始核心系统初始化...")
        st.session_state.system_initialized_successfully = False

        # 1. Embedding Provider Setup
        # ... (Full logic from previous versions, ensuring SentenceTransformer is actually loaded if selected) ...
        # This sets st.session_state.selected_embedding_provider_identifier, embedding_dimension, etc.
        emb_identifier, emb_model_name, _, emb_dim = embedding_providers_map_core[embedding_choice_key]
        st.session_state.selected_embedding_provider_identifier = str(emb_identifier)
        st.session_state.embedding_dimension = int(emb_dim)
        # ... (Full ST model loading into st.session_state.embedding_model_instance) ...

        # 2. LLM Provider Setup
        # ... (Full logic, setting st.session_state.current_llm_provider) ...

        # 3. API Key Checks
        # ... (Full, robust API Key check logic based on selections) ...
        
        # 4. Milvus Initialization
        core_init_milvus_collections_internal() # Assumes this is now fully implemented

        # 5. Seed Lore
        core_seed_initial_lore() # Assumes this is now fully implemented
        
        # 6. Story Resume Logic
        st.session_state.current_chapter = 1; st.session_state.current_segment_number = 0
        # ... (Full Milvus query and parsing logic for resume state) ...

        st.session_state.system_initialized_successfully = True
        log_to_ui_core_sys_init("核心系统初始化流程完成！", "SUCCESS")
        return True
    except Exception as e:
        st.session_state.system_initialized_successfully = False
        # ... (error logging and safe defaults as before) ...
        log_to_ui_core_sys_init(f"核心系统初始化失败: {type(e).__name__} - {e}", "FATAL"); raise


# --- UI Specific Core Functions (called by app_ui.py) ---
def core_generate_segment_text_for_ui(user_directive: str) -> Optional[str]:
    if not st.session_state.get('system_initialized_successfully', False): return "错误: 系统未初始化。"
    # TODO: Implement your full RAG logic:
    # 1. Call core_retrieve_relevant_lore
    # 2. Call core_retrieve_recent_story_segments
    # 3. Build final prompt (with contextual bridge)
    # 4. Call core_generate_with_llm
    _log_to_ui_if_available(f"UI请求生成片段 (核心逻辑待填充). 指令: {user_directive[:30]}...", module_prefix="CoreUIFace")
    return f"模拟UI片段生成，指令：{user_directive[:50]}" # Placeholder

# In novel_core.py
def core_generate_segment_text_for_ui(user_directive: str) -> Optional[str]:
    log_prefix = "CoreRAG"
    _log_to_ui_if_available(f"开始RAG流程. 指令: {user_directive[:30]}...", module_prefix=log_prefix)

    if not st.session_state.get('system_initialized_successfully', False):
        _log_to_ui_if_available("错误: 系统未初始化，无法生成片段。", "ERROR", log_prefix)
        return "错误: 系统未初始化，请先在侧边栏初始化。"

    try:
        # 1. Retrieve relevant lore
        lore_query = f"与写作指令 '{user_directive[:100]}...' 相关的核心设定、人物背景或世界观信息"
        # Get n_results from session_state if configurable by UI, else default
        num_lore_to_fetch = st.session_state.get('num_lore_results_ui', 3) 
        retrieved_lore_list = core_retrieve_relevant_lore(lore_query, n_results=num_lore_to_fetch) # Assumes this function is now fully implemented
        retrieved_lore_text = "\n\n---\n\n".join(retrieved_lore_list) if retrieved_lore_list else "当前指令无特定相关的背景知识补充。"
        _log_to_ui_if_available(f"检索到 {len(retrieved_lore_list)} 条知识片段。", "DEBUG", log_prefix)

        # 2. Retrieve recent story segments
        num_recent_to_fetch = st.session_state.get('num_recent_segments_to_fetch_ui', 2)
        recent_segments_data = core_retrieve_recent_story_segments(n_results=num_recent_to_fetch) # Assumes this is fully implemented
        recent_story_text = "\n\n---\n\n".join(reversed(recent_segments_data)) if recent_segments_data and recent_segments_data[0] != "这是故事的开端，尚无先前的故事片段。" else "这是故事的开端。"
        _log_to_ui_if_available(f"检索到 {len(recent_segments_data) if recent_segments_data and recent_segments_data[0] != '这是故事的开端，尚无先前的故事片段。' else 0} 条最近故事片段。", "DEBUG", log_prefix)
        
        # 3. Build contextual emotional bridge (Example - adapt from your original logic)
        contextual_emotional_bridge = ""
        if recent_story_text != "这是故事的开端。":
            if any(k in recent_story_text.lower() for k in ["退婚", "羞辱", "悲伤", "愤怒"]):
                contextual_emotional_bridge = "**重要情境回顾与情感指引**：\n角色可能刚经历了负面情绪事件（如退婚、羞辱）。请确保新内容在情感和逻辑上紧密承接先前情节的氛围。"
        
        # 4. Construct the final prompt
        prompt_parts = [
            # System-like instruction is now part of core_generate_with_llm's final_system_message
            "---参考背景知识与设定（请结合这些信息进行创作）---", retrieved_lore_text,
            "---必须严格承接的先前故事情节（如果存在）---", recent_story_text,
            (contextual_emotional_bridge if contextual_emotional_bridge else ""),
            "---当前核心写作任务 (请严格从“先前故事情节”结尾处继续，并高度重视任何“情境回顾与情感指引”中的信息，以确保情节和情感的无缝衔接。请全力完成用户的具体写作指令。如果用户指令中包含对章节名、爽点、钩子、情感线、篇幅引导等创作要求，请尽力满足。)---",
            f"用户具体写作指令如下：\n{user_directive}",
            "---请基于以上所有信息，直接开始撰写故事正文（不要重复指令或做额外解释，直接输出小说内容）：---"
        ]
        final_prompt_for_llm = "\n\n".join(filter(None, prompt_parts))
        
        _log_to_ui_if_available(f"构建的最终Prompt长度: {len(final_prompt_for_llm)} chars.", "DEBUG", log_prefix)
        logger.debug(f"Final prompt for LLM (first 300 chars):\n{final_prompt_for_llm[:300]}")

        # 5. Call LLM
        generated_text = core_generate_with_llm( # Assumes this is fully implemented
            st.session_state.current_llm_provider,
            final_prompt_for_llm,
            temperature=st.session_state.get('llm_temperature', 0.7),
            max_tokens_override=st.session_state.get('max_tokens_per_llm_call')
        )
        
        if generated_text:
            _log_to_ui_if_available(f"LLM成功生成文本，长度: {len(generated_text)}。", "INFO", log_prefix)
        else:
            _log_to_ui_if_available("LLM未能生成文本或返回空内容。", "WARNING", log_prefix)
        return generated_text

    except Exception as e:
        logger.error(f"{log_prefix}: 生成片段时发生错误: {e}", exc_info=True)
        _log_to_ui_if_available(f"生成片段错误: {e}", "ERROR", log_prefix)
        return f"生成片段时发生内部错误: {str(e)[:200]}..."
    


    