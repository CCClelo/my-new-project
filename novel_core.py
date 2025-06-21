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
def _create_or_get_collection(collection_name: str, schema_fields: List[FieldSchema], description: str) -> Collection:
    """创建或获取Milvus集合"""
    log_helper = lambda msg, lvl="INFO": _log_to_ui_if_available(msg, lvl, f"CollHelper_{collection_name[:10]}")
    log_helper(f"开始处理集合 '{collection_name}'...")
    
    try:
        # 检查集合是否存在
        if utility.has_collection(collection_name, using=MILVUS_ALIAS_CORE):
            log_helper(f"集合 '{collection_name}' 已存在，正在获取...")
            collection = Collection(collection_name, using=MILVUS_ALIAS_CORE)
            log_helper(f"成功获取集合 '{collection_name}'")
            return collection
        
        log_helper(f"集合 '{collection_name}' 不存在，正在创建...")
        
        # 验证schema
        if not schema_fields:
            error_msg = f"创建集合 '{collection_name}' 时schema不能为空"
            log_helper(error_msg, "ERROR")
            raise ValueError(error_msg)
            
        # 创建schema和集合
        schema = CollectionSchema(
            fields=schema_fields,
            description=description,
            enable_dynamic_field=True
        )
        collection = Collection(
            name=collection_name,
            schema=schema,
            using=MILVUS_ALIAS_CORE
        )
        log_helper(f"集合 '{collection_name}' 创建成功")
        
        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 32, "efConstruction": 256}
        }
        log_helper(f"为集合 '{collection_name}' 创建索引...")
        collection.create_index(
            field_name="embedding",
            index_params=index_params,
            index_name=f"idx_{collection_name}",
            timeout=180,
            using=MILVUS_ALIAS_CORE
        )
        log_helper(f"集合 '{collection_name}' 索引创建成功")
        
        # 加载集合
        log_helper(f"加载集合 '{collection_name}' 到内存...")
        collection.load(timeout=180, using=MILVUS_ALIAS_CORE)
        log_helper(f"集合 '{collection_name}' 加载完成")
        
        return collection
        if utility.has_collection(collection_name):
            _log_to_ui_if_available(f"集合 {collection_name} 已存在，直接获取", "DEBUG", "CoreMilvus")
            return Collection(collection_name)
        
        _log_to_ui_if_available(f"创建新集合: {collection_name}", "INFO", "CoreMilvus")
        schema = CollectionSchema(
            fields=schema_fields,
            description=description,
            enable_dynamic_field=True
        )
        
        collection = Collection(
            name=collection_name,
            schema=schema,
            using=MILVUS_ALIAS_CORE
        )
        
        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024}
        }
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        return collection
    except Exception as e:
        logger.error(f"创建/获取集合 {collection_name} 失败: {e}", exc_info=True)
        raise

def core_load_and_vectorize_settings():
    """加载并向量化设定文件到Milvus知识库"""
    log_prefix = "CoreLoadSettings"
    _log_to_ui_if_available("开始加载和向量化设定文件...", module_prefix=log_prefix)
    
    if not st.session_state.get('milvus_initialized_core', False) or \
       not isinstance(st.session_state.get('lore_collection_milvus_obj'), Collection):
        logger.error("Core: Milvus lore collection not properly initialized for loading settings.")
        _log_to_ui_if_available("错误: 知识库Milvus集合未初始化。", "ERROR", log_prefix)
        return

    lore_collection: Collection = st.session_state.lore_collection_milvus_obj
    force_reload = os.environ.get("FORCE_RELOAD_SETTINGS", "false").lower() == "true"

    try:
        current_entities = lore_collection.num_entities
        logger.info(f"Core: Current entities in '{lore_collection.name}': {current_entities}")
    except Exception as e_num_entities:
        logger.error(f"Core: Failed to get num_entities for '{lore_collection.name}': {e_num_entities}. Assuming 0 for reload check.")
        current_entities = 0

    if not force_reload and current_entities > 0:
        logger.info(f"Core: Knowledge base '{lore_collection.name}' not empty ({current_entities} entities) and FORCE_RELOAD_SETTINGS is false. Skipping load.")
        _log_to_ui_if_available(f"知识库 '{lore_collection.name}' 已有内容 ({current_entities} 条)，跳过加载。", "INFO", log_prefix)
        return
    elif force_reload:
        logger.info(f"Core: FORCE_RELOAD_SETTINGS is true. Will attempt to reload settings into '{lore_collection.name}'.")
        _log_to_ui_if_available("强制重新加载知识库...", "WARNING", log_prefix)

    settings_dir = SETTINGS_FILES_DIR_CORE
    if not os.path.exists(settings_dir):
        logger.warning(f"Core: Settings directory '{settings_dir}' not found.")
        _log_to_ui_if_available(f"警告: 设定文件目录 '{settings_dir}' 未找到。", "WARNING", log_prefix)
        return

    files_processed, chunks_added_total = 0, 0
    batch_size_embed = st.session_state.get("embed_batch_size", 32)
    batch_size_insert = st.session_state.get("milvus_insert_batch_size", 100)
    all_entities_to_insert = []

    _log_to_ui_if_available(f"开始从 '{settings_dir}' 读取文件...", "DEBUG", log_prefix)
    for filename in os.listdir(settings_dir):
        if filename.endswith((".txt", ".md")):
            filepath = os.path.join(settings_dir, filename)
            _log_to_ui_if_available(f"处理文件: {filename}...", "DEBUG", log_prefix)
            logger.info(f"Core: Processing settings file: {filepath}")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f: 
                    content = f.read()
                
                text_chunks_from_file = core_chunk_text_by_paragraph(content)
                if not text_chunks_from_file:
                    logger.info(f"Core: File '{filename}' is empty or has no valid paragraphs.")
                    continue

                _log_to_ui_if_available(f"文件 '{filename}' 分块完成，共 {len(text_chunks_from_file)} 块。", "DEBUG", log_prefix)
                file_type_tag = os.path.splitext(filename)[0].lower().replace(' ', '_').replace('-', '_')

                for i in range(0, len(text_chunks_from_file), batch_size_embed):
                    batch_texts = text_chunks_from_file[i:i+batch_size_embed]
                    _log_to_ui_if_available(f"为 '{filename}' 的批次 {i//batch_size_embed + 1} (大小 {len(batch_texts)}) 生成向量...", "DEBUG", log_prefix)
                    
                    vectors = None
                    embedding_provider_id = st.session_state.selected_embedding_provider_identifier

                    if embedding_provider_id == "openai_official":
                        vectors = core_get_openai_embeddings(batch_texts, OPENAI_EMBEDDING_MODEL_CORE)
                    elif embedding_provider_id.startswith("sentence_transformer_"):
                        vectors = core_get_st_embeddings(batch_texts)

                    if not vectors or len(vectors) != len(batch_texts):
                        logger.error(f"Core: Failed to generate embeddings for a batch from '{filename}'. Skipping this batch.")
                        _log_to_ui_if_available(f"文件 '{filename}' 部分嵌入生成失败，跳过批次。", "ERROR", log_prefix)
                        continue

                    for j, chunk_text in enumerate(batch_texts):
                        chunk_hash = uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_{i+j}_{chunk_text[:128]}").hex[:16]
                        doc_id = f"setting_{file_type_tag}_{chunk_hash}"

                        all_entities_to_insert.append({
                            "doc_id": doc_id, 
                            "embedding": vectors[j], 
                            "text_content": chunk_text,
                            "timestamp": datetime.now().isoformat(), 
                            "source_file": filename,
                            "document_type": file_type_tag
                        })

                files_processed += 1
            except Exception as e_file_proc:
                logger.error(f"Core: Error processing settings file '{filename}': {e_file_proc}", exc_info=True)
                _log_to_ui_if_available(f"处理文件 '{filename}' 失败: {e_file_proc}", "ERROR", log_prefix)

    if all_entities_to_insert:
        _log_to_ui_if_available(f"准备将 {len(all_entities_to_insert)} 个设定片段插入Milvus...", "INFO", log_prefix)
        for i in range(0, len(all_entities_to_insert), batch_size_insert):
            batch_entities = all_entities_to_insert[i:i+batch_size_insert]
            try:
                logger.info(f"Core: Inserting batch of {len(batch_entities)} entities into lore collection '{lore_collection.name}'.")
                insert_result = lore_collection.insert(batch_entities)
                chunks_added_total += len(insert_result.primary_keys)
                _log_to_ui_if_available(f"成功插入 {len(insert_result.primary_keys)} 个设定片段 (批次 {i//batch_size_insert + 1})。", "DEBUG", log_prefix)
            except Exception as e_insert:
                logger.error(f"Core: Milvus insert failed for settings batch: {e_insert}", exc_info=True)
                _log_to_ui_if_available(f"Milvus设定片段插入失败 (批次 {i//batch_size_insert + 1}): {e_insert}", "ERROR", log_prefix)

        if chunks_added_total > 0:
            try:
                logger.info(f"Core: Flushing lore collection '{lore_collection.name}' after inserting {chunks_added_total} entities.")
                lore_collection.flush(timeout=60)
                _log_to_ui_if_available(f"Milvus知识库 '{lore_collection.name}' 已刷新。", "INFO", log_prefix)
            except Exception as e_flush:
                logger.error(f"Core: Milvus flush failed for lore collection: {e_flush}", exc_info=True)
                _log_to_ui_if_available(f"Milvus知识库刷新失败: {e_flush}", "ERROR", log_prefix)

    logger.info(f"Core: Finished loading settings. Processed {files_processed} files, total chunks added this session: {chunks_added_total}.")
    _log_to_ui_if_available(f"设定文件加载完成。处理 {files_processed} 文件，共添加 {chunks_added_total} 片段。", "INFO", log_prefix)

def core_init_milvus_collections_internal():
    """初始化Milvus集合"""
    log_prefix = "CoreInitMilvus"
    log_to_ui_milvus = lambda msg, lvl="INFO": _log_to_ui_if_available(msg, lvl, log_prefix)
    logger.info("Core: core_init_milvus_collections_internal called for ACTUAL collection setup.")
    log_to_ui_milvus("开始实际的Milvus集合初始化...")

    try:
        # 检查前置条件
        if 'embedding_dimension' not in st.session_state:
            raise ValueError("embedding_dimension 未在session_state中设置")
            
        embedding_dim = st.session_state.embedding_dimension
        provider_id = st.session_state.selected_embedding_provider_identifier
        
        # 建立连接
        zilliz_uri_from_env = os.getenv(ZILLIZ_CLOUD_URI_ENV_NAME)
        zilliz_token_from_env = os.getenv(ZILLIZ_CLOUD_TOKEN_ENV_NAME)

        # 添加调试日志
        logger.info(f"DEBUG MilvusConnect: Read ZILLIZ_CLOUD_URI from env: '{zilliz_uri_from_env}'")
        token_present_for_log = "IS SET and NOT EMPTY" if zilliz_token_from_env and zilliz_token_from_env.strip() else "IS NOT SET or EMPTY"
        logger.info(f"DEBUG MilvusConnect: Read ZILLIZ_CLOUD_TOKEN from env: Status is '{token_present_for_log}'")
        if zilliz_token_from_env:
            logger.info(f"DEBUG MilvusConnect: ZILLIZ_CLOUD_TOKEN (first 5 chars if set): '{str(zilliz_token_from_env)[:5]}...'")
        
        log_to_ui_milvus(f"Env Check: URI='{zilliz_uri_from_env}', Token Status='{token_present_for_log}'")

        alias_to_use_in_connect = MILVUS_ALIAS_CORE
        try:
            if connections.has_connection(alias_to_use_in_connect):
                connections.remove_connection(alias_to_use_in_connect)

            # 改进的Zilliz Cloud连接条件判断
            use_zilliz = (
                zilliz_uri_from_env and zilliz_uri_from_env.strip() and 
                zilliz_token_from_env and zilliz_token_from_env.strip() and
                zilliz_uri_from_env != "your_zilliz_cluster_uri_from_screenshot" and
                zilliz_token_from_env != "YOUR_ACTUAL_ZILLIZ_CLOUD_TOKEN_HERE"
            )
            logger.info(f"DEBUG MilvusConnect: Condition to use Zilliz Cloud is: {use_zilliz}")

            if use_zilliz: 
                logger.info(f"Core: Attempting Zilliz Cloud connection with URI: {zilliz_uri_from_env}")
                connections.connect(alias=alias_to_use_in_connect, uri=zilliz_uri_from_env, token=zilliz_token_from_env, timeout=30.0)
                st.session_state.milvus_target = "Zilliz Cloud"
            else:
                logger.info(f"Core: Zilliz Cloud config incomplete, placeholder, or not set. Attempting local Milvus: {MILVUS_HOST_CORE}:{MILVUS_PORT_CORE}")
                connections.connect(alias=alias_to_use_in_connect, host=MILVUS_HOST_CORE, port=MILVUS_PORT_CORE, timeout=10.0)
                st.session_state.milvus_target = "Local"
            
            logger.info(f"Core: Milvus connected (Target: {st.session_state.get('milvus_target', '未知')}).")
            log_to_ui_milvus(f"Milvus连接成功 (目标: {st.session_state.get('milvus_target', '未知')}).", "SUCCESS")
        except Exception as e:
            logger.error(f"Core: Milvus连接失败 during connect(): {e}", exc_info=True)
            log_to_ui_milvus(f"Milvus连接失败: {e}", "ERROR")
            raise 

        # 定义集合名称
        lore_col_name = f"{COLLECTION_NAME_LORE_PREFIX_CORE}_{provider_id}_{embedding_dim}d"
        story_col_name = f"{COLLECTION_NAME_STORY_PREFIX_CORE}_{provider_id}_{embedding_dim}d"
        
        # 定义schema
        lore_schema_list = [
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=64)
        ]
        
        story_schema_list = [
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chapter", dtype=DataType.INT64),
            FieldSchema(name="segment_number", dtype=DataType.INT64),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="user_directive", dtype=DataType.VARCHAR, max_length=2048)
        ]
        
        # 创建/获取集合
        log_to_ui_milvus("开始获取/创建Lore Collection...")
        st.session_state.lore_collection_milvus_obj = _create_or_get_collection(
            lore_col_name,
            lore_schema_list,
            "Novel Lore Knowledge Collection"
        )
        st.session_state.lore_collection_name = lore_col_name
        log_to_ui_milvus("Lore Collection准备就绪")
        
        log_to_ui_milvus("开始获取/创建Story Collection...")
        st.session_state.story_collection_milvus_obj = _create_or_get_collection(
            story_col_name,
            story_schema_list,
            "Novel Story Segments Collection"
        )
        st.session_state.story_collection_name = story_col_name
        log_to_ui_milvus("Story Collection准备就绪")
        
        st.session_state.milvus_initialized_core = True
        logger.info(f"Milvus collections '{lore_col_name}' and '{story_col_name}' are fully ready")
        log_to_ui_milvus("所有Milvus集合准备就绪", "SUCCESS")
        
    except Exception as e:
        st.session_state.milvus_initialized_core = False
        logger.error(f"初始化Milvus集合失败: {e}", exc_info=True)
        log_to_ui_milvus(f"初始化Milvus集合失败: {e}", "ERROR")
        raise
        # 检查前置条件
        if 'embedding_dimension' not in st.session_state:
            raise ValueError("embedding_dimension 未在session_state中设置")
            
        embedding_dim = st.session_state.embedding_dimension
        provider_id = st.session_state.selected_embedding_provider_identifier
        
        # 建立连接
        _log_to_ui_if_available("正在连接Milvus...", module_prefix=log_prefix)
        connections.connect(
            alias=MILVUS_ALIAS_CORE,
            host=MILVUS_HOST_CORE,
            port=MILVUS_PORT_CORE
        )
        
        # 定义集合名称
        lore_col_name = f"{COLLECTION_NAME_LORE_PREFIX_CORE}_{provider_id}_{embedding_dim}d"
        story_col_name = f"{COLLECTION_NAME_STORY_PREFIX_CORE}_{provider_id}_{embedding_dim}d"
        
        # 定义schema
        lore_schema_list = [
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=64)
        ]
        
        story_schema_list = [
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chapter", dtype=DataType.INT64),
            FieldSchema(name="segment_number", dtype=DataType.INT64),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="user_directive", dtype=DataType.VARCHAR, max_length=2048)
        ]
        
        # 创建/获取集合
        _log_to_ui_if_available("开始获取/创建Lore Collection...", module_prefix=log_prefix)
        st.session_state.lore_collection_milvus_obj = _create_or_get_collection(
            lore_col_name,
            lore_schema_list,
            "Novel Lore Knowledge Collection"
        )
        st.session_state.lore_collection_name = lore_col_name
        _log_to_ui_if_available("Lore Collection准备就绪", module_prefix=log_prefix)
        
        _log_to_ui_if_available("开始获取/创建Story Collection...", module_prefix=log_prefix)
        st.session_state.story_collection_milvus_obj = _create_or_get_collection(
            story_col_name,
            story_schema_list,
            "Novel Story Segments Collection"
        )
        st.session_state.story_collection_name = story_col_name
        _log_to_ui_if_available("Story Collection准备就绪", module_prefix=log_prefix)
        
        st.session_state.milvus_initialized_core = True
        logger.info(f"Milvus collections '{lore_col_name}' and '{story_col_name}' are fully ready")
        _log_to_ui_if_available("所有Milvus集合准备就绪", "SUCCESS", log_prefix)
        
    except Exception as e:
        st.session_state.milvus_initialized_core = False
        logger.error(f"初始化Milvus集合失败: {e}", exc_info=True)
        _log_to_ui_if_available(f"初始化Milvus集合失败: {e}", "ERROR", log_prefix)
        raise

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

def core_generate_with_llm(provider_name: Optional[str], # Make it Optional
                           prompt_text_from_rag: str, 
                           temperature: float =0.7, 
                           max_tokens_override: Optional[int]=None, 
                           system_message_override: Optional[str]=None) -> Optional[str]:
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
        log_to_ui_core_sys_init("步骤2: 设置LLM提供商...")
        
        # Validate LLM choice key
        if not llm_choice_key or not isinstance(llm_choice_key, str) or llm_choice_key not in llm_providers_map_core:
            err_msg_llm_key = f"无效的 llm_choice_key: '{llm_choice_key}'. 可用: {list(llm_providers_map_core.keys())}"
            logger.error(err_msg_llm_key)
            log_to_ui_core_sys_init(err_msg_llm_key, "FATAL")
            raise ValueError(err_msg_llm_key)
            
        llm_provider_name = llm_providers_map_core[llm_choice_key]
        
        # Validate provider name from map
        if not llm_provider_name or not isinstance(llm_provider_name, str):
            err_msg_llm_val = f"llm_providers_map_core 中 key '{llm_choice_key}' 对应的值 ('{llm_provider_name}') 无效."
            logger.error(err_msg_llm_val)
            log_to_ui_core_sys_init(err_msg_llm_val, "FATAL")
            raise ValueError(err_msg_llm_val)
            
        st.session_state.current_llm_provider = llm_provider_name
        logger.info(f"Core: current_llm_provider SET in session_state to: '{st.session_state.current_llm_provider}'")

        # 3. API Key Checks
        # ... (Full, robust API Key check logic based on selections) ...
        
        # 4. Milvus Initialization
        log_to_ui_core_sys_init("步骤4: 初始化Milvus集合...")
        try:
            core_init_milvus_collections_internal()
            if not st.session_state.get('milvus_initialized_core', False):
                raise RuntimeError("Milvus集合初始化失败")
            log_to_ui_core_sys_init("Milvus集合初始化成功", "DEBUG")
        except Exception as e_milvus:
            logger.error(f"Core: Milvus初始化失败: {e_milvus}", exc_info=True)
            log_to_ui_core_sys_init(f"Milvus初始化失败: {e_milvus}", "FATAL")
            raise

        # 5. Seed Lore Data
        log_to_ui_core_sys_init("步骤5: 加载种子数据...")
        try:
            core_seed_initial_lore()
            lore_count = st.session_state.lore_collection_milvus_obj.num_entities
            log_to_ui_core_sys_init(f"知识库种子数据加载完成，共{lore_count}条记录", "DEBUG")
        except Exception as e_seed:
            logger.error(f"Core: 种子数据加载失败: {e_seed}", exc_info=True)
            log_to_ui_core_sys_init(f"种子数据加载失败: {e_seed}", "ERROR")
            raise

        # 6. Story Resume Logic
        log_to_ui_core_sys_init("步骤6: 初始化故事状态...")
        try:
            st.session_state.current_chapter = 1
            st.session_state.current_segment_number = 0
            
            # Check for existing story segments
            if st.session_state.story_collection_milvus_obj.num_entities > 0:
                recent_segments = core_retrieve_recent_story_segments(n_results=1)
                if recent_segments:
                    st.session_state.last_adopted_segment_text = recent_segments[0]
                    log_to_ui_core_sys_init("检测到已有故事片段，已加载最新片段", "INFO")
            
            log_to_ui_core_sys_init("故事状态初始化完成", "DEBUG")
        except Exception as e_story:
            logger.error(f"Core: 故事状态初始化失败: {e_story}", exc_info=True)
            log_to_ui_core_sys_init(f"故事状态初始化失败: {e_story}", "WARNING")

        st.session_state.system_initialized_successfully = True
        log_to_ui_core_sys_init("核心系统初始化流程完成！", "SUCCESS")
        return True
    except Exception as e:
        st.session_state.system_initialized_successfully = False
        # ... (error logging and safe defaults as before) ...
        log_to_ui_core_sys_init(f"核心系统初始化失败: {type(e).__name__} - {e}", "FATAL"); raise


# --- UI Specific Core Functions (called by app_ui.py) ---
def core_generate_with_llm(provider_name: Optional[str], 
                         prompt_text_from_rag: str,
                         temperature: float = 0.7,
                         max_tokens_override: Optional[int] = None,
                         system_message_override: Optional[str] = None) -> Optional[str]:
    if not provider_name or not isinstance(provider_name, str):
        logger.error(f"CoreLLM CRITICAL: provider_name is invalid or None: '{provider_name}'")
        return "错误：LLM提供商名称无效或未设置。"
    
    log_prefix = f"CoreLLM_{provider_name.upper()}"
    _log_to_ui_if_available(f"开始调用LLM: {provider_name}...", module_prefix=log_prefix)

    # Default system message if not overridden
    final_system_message = system_message_override or """
    你是一位专业的小说创作助手，擅长根据用户指令和上下文创作高质量的小说内容。
    请严格遵循以下规则：
    1. 保持角色性格和世界观一致性
    2. 自然承接前文情节
    3. 直接输出小说正文，不要添加解释或注释
    """

    try:
        if provider_name == "openai_official":
            api_key = st.session_state.get('api_keys', {}).get(OPENAI_API_KEY_ENV_NAME, OPENAI_API_KEY_CORE)
            if not api_key:
                raise ValueError("OpenAI API Key未配置")
            
            client = openai.OpenAI(
                api_key=api_key,
                http_client=core_get_httpx_client_with_proxy(
                    OPENAI_OFFICIAL_HTTP_PROXY_CORE,
                    OPENAI_OFFICIAL_HTTPS_PROXY_CORE
                )
            )
            
            response = client.chat.completions.create(
                model=OPENAI_LLM_MODEL_CORE,
                messages=[
                    {"role": "system", "content": final_system_message},
                    {"role": "user", "content": prompt_text_from_rag}
                ],
                temperature=temperature,
                max_tokens=max_tokens_override or 2000
            )
            return response.choices[0].message.content

        elif provider_name == "deepseek":
            # Similar implementation for DeepSeek
            return "DeepSeek实现待完成"

        elif provider_name == "gemini":
            # Get API key from session state or environment
            api_key = st.session_state.get('api_keys', {}).get(GEMINI_API_KEY_ENV_NAME, os.getenv(GEMINI_API_KEY_ENV_NAME))
            if not api_key:
                raise ValueError("Gemini API Key未配置")

            # Set and log proxies
            http_proxy_to_set, https_proxy_to_set = GEMINI_HTTP_PROXY_CORE, GEMINI_HTTPS_PROXY_CORE
            logger.info(f"{log_prefix}: 正在设置系统代理: HTTP='{http_proxy_to_set}', HTTPS='{https_proxy_to_set}'")
            core_set_temp_os_proxies(http_proxy_to_set, https_proxy_to_set)
            
            # Debug log actual proxy values
            logger.debug(f"{log_prefix}: 当前系统代理设置: HTTP_PROXY='{os.environ.get('HTTP_PROXY')}', HTTPS_PROXY='{os.environ.get('HTTPS_PROXY')}'")
            _log_to_ui_if_available(
                f"Gemini调用前代理设置: HTTP='{os.environ.get('HTTP_PROXY')}', HTTPS='{os.environ.get('HTTPS_PROXY')}'",
                "DEBUG",
                log_prefix
            )

            # Check if client needs re-initialization
            gemini_client = st.session_state.get('gemini_llm_client_core')
            if not gemini_client or st.session_state.get('current_gemini_client_provider_details') != provider_name:
                # Configure Gemini SDK if not already done
                if not st.session_state.get('gemini_configured_core_sdk_v2', False):
                    logger.info(f"{log_prefix}: Configuring Gemini SDK")
                    
                    # Prepare client options
                    client_options = None
                    api_endpoint = os.getenv("GEMINI_API_ENDPOINT")  # e.g. for Vertex AI
                    if api_endpoint:
                        client_options = {"api_endpoint": api_endpoint}
                        logger.debug(f"{log_prefix}: Using custom API endpoint: {api_endpoint}")

                    try:
                        genai.configure(
                            api_key=api_key,
                            client_options=client_options
                        )
                        st.session_state.gemini_configured_core_sdk_v2 = True
                        logger.info(f"{log_prefix}: Gemini SDK configured successfully")
                    except Exception as config_error:
                        logger.error(f"{log_prefix}: Gemini SDK配置失败: {config_error}", exc_info=True)
                        raise ValueError(f"Gemini SDK配置失败: {config_error}")
                
                # Initialize new client
                logger.info(f"{log_prefix}: Initializing Gemini GenerativeModel for {GEMINI_LLM_MODEL_CORE}")
                gemini_client = genai.GenerativeModel(
                    GEMINI_LLM_MODEL_CORE,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens_override or 2000,
                        temperature=temperature
                    )
                )
                st.session_state.gemini_llm_client_core = gemini_client
                st.session_state.current_gemini_client_provider_details = provider_name
            
            # Prepare prompt with system message
            prompt_with_sys_msg = f"{final_system_message}\n\n---\n\n{prompt_text_from_rag}"
            
            # Configure safety settings
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
            
            logger.info(f"{log_prefix}: Sending prompt to Gemini (length: {len(prompt_with_sys_msg)})")
            _log_to_ui_if_available("正在调用Gemini API...", "DEBUG", log_prefix)
            
            try:
                response = gemini_client.generate_content(
                    prompt_with_sys_msg,
                    safety_settings=safety_settings,
                    generation_config={
                        'temperature': temperature,
                        'max_output_tokens': max_tokens_override or 2000
                    }
                )
                
                # Process response
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason
                    logger.warning(f"{log_prefix}: Gemini Prompt Blocked: {block_reason}")
                    _log_to_ui_if_available(f"Gemini内容被阻止: {block_reason}", "ERROR", log_prefix)
                    return f"错误：Gemini内容由于'{block_reason}'而被阻止。"
                
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    generated_text = "".join(part.text for part in response.candidates[0].content.parts).strip()
                    if not generated_text:
                        logger.warning(f"{log_prefix}: Gemini parts joined to empty string")
                        return "Gemini返回了空内容"
                    return generated_text
                
                logger.warning(f"{log_prefix}: Gemini returned no parsable content")
                return "Gemini未能返回有效内容"
                
            except Exception as e:
                logger.error(f"{log_prefix}: Gemini调用失败: {e}", exc_info=True)
                _log_to_ui_if_available(f"Gemini调用失败: {e}", "ERROR", log_prefix)
                return f"Gemini生成错误: {str(e)[:200]}"

        elif provider_name == "custom_proxy_llm":
            # Similar implementation for custom proxy
            return "Custom Proxy实现待完成"

        else:
            logger.error(f"未知的LLM提供商: {provider_name}")
            return f"错误：不支持的LLM提供商 '{provider_name}'"

    except Exception as e:
        logger.error(f"{log_prefix}: LLM调用失败: {e}", exc_info=True)
        _log_to_ui_if_available(f"LLM调用失败: {e}", "ERROR", log_prefix)
        return f"LLM生成错误: {str(e)[:200]}"

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

        # 5. Validate LLM provider before calling
        current_llm_provider_from_state = st.session_state.get('current_llm_provider')
        if not current_llm_provider_from_state or not isinstance(current_llm_provider_from_state, str):
            err_msg_llm_missing = "错误 (CoreRAG): current_llm_provider 未在session_state中正确设置，无法调用LLM。"
            logger.error(err_msg_llm_missing)
            _log_to_ui_if_available(err_msg_llm_missing, "ERROR", log_prefix)
            return err_msg_llm_missing

        # 6. Call LLM with validated provider
        generated_text = core_generate_with_llm(
            current_llm_provider_from_state,
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
    


    