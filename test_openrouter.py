import os
import unittest
from unittest.mock import patch, MagicMock

# Force configuration env variables for testing offline
os.environ["STORAGE_TYPE"] = "local"
os.environ["LOCAL_STORAGE_DIR"] = "./data/storage"

from app.core.config import settings
from app.services.openrouter_service import openrouter_service, OpenRouterService

class TestOpenRouterIntegration(unittest.TestCase):
    
    def setUp(self):
        self.original_key = settings.OPENROUTER_API_KEY

    def tearDown(self):
        settings.OPENROUTER_API_KEY = self.original_key
        
    def test_settings_loading(self):
        """Verify OpenRouter configurations are loaded correctly."""
        self.assertIsNotNone(settings.OPENROUTER_MODEL)
        self.assertEqual(settings.OPENROUTER_MODEL, "google/gemini-2.5-flash")

    def test_headers_fails_without_key(self):
        """Verify headers build raises ValueError if API Key is placeholder or missing."""
        service = OpenRouterService()
        
        # Test with empty key
        settings.OPENROUTER_API_KEY = ""
        with self.assertRaises(ValueError) as context:
            service._get_headers()
        self.assertIn("API key is not configured", str(context.exception))
            
        # Test with default placeholder
        settings.OPENROUTER_API_KEY = "your_openrouter_api_key_here"
        with self.assertRaises(ValueError) as context:
            service._get_headers()
        self.assertIn("API key is not configured", str(context.exception))

    @patch('httpx.Client')
    def test_mock_chat_completion(self, mock_client_class):
        """Mock OpenRouter API call and verify requests payload structure."""
        service = OpenRouterService()
        
        # Mock httpx responses
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "AI design analysis: 5 colors, floral motifs, straight repeat."
                    }
                }
            ]
        }
        mock_client.post.return_value = mock_response
        # Return mock_client from with statement context manager
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        settings.OPENROUTER_API_KEY = "valid_mock_key"
        # Trigger call
        messages = [{"role": "user", "content": "Analyze image"}]
        result = service.call_chat_completion(messages)
        
        # Assert result content
        self.assertEqual(
            result["choices"][0]["message"]["content"], 
            "AI design analysis: 5 colors, floral motifs, straight repeat."
        )
        
        # Assert headers and endpoint were correct
        mock_client.post.assert_called_once()
        called_args, called_kwargs = mock_client.post.call_args
        self.assertEqual(called_args[0], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(called_kwargs["headers"]["Authorization"], "Bearer valid_mock_key")
        self.assertEqual(called_kwargs["json"]["model"], "google/gemini-2.5-flash")
        self.assertEqual(called_kwargs["json"]["max_tokens"], 4096)

    @patch('httpx.Client')
    def test_model_override_uses_openrouter(self, mock_client_class):
        """Verify a requested OpenRouter model is sent to OpenRouter unchanged."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        settings.OPENROUTER_API_KEY = "valid_mock_key"
        OpenRouterService().call_chat_completion(
            [{"role": "user", "content": "Hello"}],
            model="openrouter/auto",
        )

        called_args, called_kwargs = mock_client.post.call_args
        self.assertEqual(called_args[0], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(called_kwargs["json"]["model"], "openrouter/auto")

    @patch('httpx.Client')
    def test_mock_design_analysis(self, mock_client_class):
        """Mock Vision API design analysis."""
        service = OpenRouterService()
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Vision Model Response Content"
                    }
                }
            ]
        }
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        settings.OPENROUTER_API_KEY = "valid_mock_key"
        content = service.analyze_design("base64data", mime_type="image/png")
        self.assertEqual(content, "Vision Model Response Content")
        
        # Check Vision payload layout
        called_args, called_kwargs = mock_client.post.call_args
        payload_messages = called_kwargs["json"]["messages"]
        user_content = payload_messages[0]["content"]
        
        # Message contains prompt text and image block
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertEqual(user_content[1]["image_url"]["url"], "data:image/png;base64,base64data")

if __name__ == "__main__":
    unittest.main()
