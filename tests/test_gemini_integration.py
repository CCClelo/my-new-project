"""Gemini API 集成测试脚本"""
import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GeminiTest")

def test_gemini_integration():
    """测试Gemini API集成"""
    try:
        # 加载环境变量
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        GEMINI_API_ENDPOINT = os.getenv("GEMINI_API_ENDPOINT")
        
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY 未在.env文件中设置")
            return False

        # 配置客户端
        logger.info("配置GenAI客户端...")
        client_options = {"api_endpoint": GEMINI_API_ENDPOINT} if GEMINI_API_ENDPOINT else None
        genai.configure(
            api_key=GEMINI_API_KEY,
            client_options=client_options
        )

        # 初始化模型
        logger.info("初始化GenerativeModel...")
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # 测试生成内容
        test_prompt = "你好，Gemini！请用中文写一首关于春天的短诗。"
        logger.info(f"发送测试提示: {test_prompt}")
        
        response = model.generate_content(test_prompt)
        logger.info("收到响应")

        # 处理响应
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            logger.error(f"提示被阻止: {response.prompt_feedback.block_reason}")
            return False

        generated_text = None
        if response.candidates and response.candidates[0].content.parts:
            generated_text = "".join(part.text for part in response.candidates[0].content.parts).strip()
        elif hasattr(response, 'text'):
            generated_text = response.text

        if generated_text:
            logger.info("生成内容成功:\n%s", generated_text)
            return True
        else:
            logger.error("未能获取有效响应内容")
            return False

    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    if test_gemini_integration():
        print("✅ Gemini集成测试通过")
    else:
        print("❌ Gemini集成测试失败")
