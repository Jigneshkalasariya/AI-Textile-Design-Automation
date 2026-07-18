import httpx
from typing import Dict, Any, List, Optional
from app.core.config import settings
from app.core.logger import logger
from app.models.request_models import DEFAULT_CAD_PROMPT

class OpenRouterService:
    """
    Service to interact with the OpenRouter API (https://openrouter.ai/).
    Supports both text chat completions and multimodal vision inputs to analyze artwork.
    """
    
    @property
    def api_key(self) -> str:
        return settings.OPENROUTER_API_KEY

    @property
    def default_model(self) -> str:
        return settings.OPENROUTER_MODEL or "google/gemini-2.5-flash"

    def _get_headers(self) -> Dict[str, str]:
        if not self.api_key or self.api_key == "your_openrouter_api_key_here":
            raise ValueError(
                "OpenRouter API key is not configured. Please set the OPENROUTER_API_KEY variable in your .env file."
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://jacquardai.com",
            "X-Title": "Jacquard AI Textile Platform"
        }

    def call_chat_completion(self, messages: List[Dict[str, Any]], model: Optional[str] = None) -> Dict[str, Any]:
        """
        Sends a chat completion request to the OpenRouter API.
        """
        target_model = model or self.default_model
        headers = self._get_headers()
        payload = {
            "model": target_model,
            "messages": messages,
            "max_tokens": settings.OPENROUTER_MAX_TOKENS,
        }
        
        logger.info(f"Sending request to OpenRouter using model: {target_model}")
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    logger.error(f"OpenRouter API returned error {response.status_code}: {response.text}")
                    response.raise_for_status()
                    
                data = response.json()
                logger.info("OpenRouter response received successfully.")
                return data
                
        except Exception as e:
            logger.exception(f"Failed to communicate with OpenRouter: {e}")
            raise

    def analyze_design(self, image_base64: str, mime_type: str = "image/png", model: Optional[str] = None) -> str:
        """
        Sends a textile design image to a Vision-LLM via OpenRouter for detailed analysis
        guided by the Antigravity Textile CAD Preprocessing Prompt.
        """
        prompt = (
            "You are a professional Textile CAD image preprocessing engine design assistant. "
            "Analyze this artwork and provide structured recommendations. Here are your guidelines:\n\n"
            f"{DEFAULT_CAD_PROMPT}\n\n"
            "Based on the guidelines above, analyze the provided design and report:\n"
            "1. Motif & Design Structure (identify flowers, borders, repeats, alignment)\n"
            "2. Defects Detected (jpeg noise, scanning noise, dust, scratches, perspective skew)\n"
            "3. Palette Analysis (approximate color count, faded colors, gamut limits)\n"
            "4. Technical Preprocessing Recommendations (adjusting denoise strength, sharpness, or colors for Texcelle import)"
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        response_data = self.call_chat_completion(messages=messages, model=model)
        try:
            choices = response_data.get("choices", [])
            if not choices:
                raise ValueError("No response choices returned by OpenRouter.")
            return choices[0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse OpenRouter response format: {e}")
            raise ValueError(f"Invalid response format from OpenRouter: {response_data}")

openrouter_service = OpenRouterService()
