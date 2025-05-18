#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenRouter Image Processing Client

This script demonstrates how to send image data to OpenRouter API
and process the response.
"""

import os
import base64
import requests
from pathlib import Path
import argparse
from typing import Optional, Dict, Any, Union
import json
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class OpenRouterClient:
    """Client for interacting with OpenRouter API to process images."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If None, will try to load from OPENROUTER_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set it as an argument or as OPENROUTER_API_KEY environment variable.")
    
    def encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Encode an image file to base64.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def process_image(
        self, 
        image_path: Union[str, Path], 
        model: str = "anthropic/claude-3-haiku-20240307",
        prompt: str = "Опиши подробно, что ты видишь на этом изображении. Ответ должен быть на русском языке.",
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Send an image to OpenRouter and get the response.
        
        Args:
            image_path: Path to the image file
            model: Model identifier to use
            prompt: Text prompt to accompany the image
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            API response as a dictionary
        """
        try:
            encoded_image = self.encode_image(image_path)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [
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
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens
            }
            
            logger.info(f"Sending request to OpenRouter API using model: {model}")
            logger.info(f"Payload structure: {json.dumps(payload, default=str, ensure_ascii=False)[:500]}...")
            
            response = requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                error_detail = ""
                try:
                    error_json = response.json()
                    error_detail = f"\nError details: {json.dumps(error_json, indent=2, ensure_ascii=False)}"
                except:
                    error_detail = f"\nResponse text: {response.text}"
                
                logger.error(f"API Error: {response.status_code}{error_detail}")
                response.raise_for_status()
            
            return response.json()
            
        except FileNotFoundError as e:
            logger.error(f"File error: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Send an image to OpenRouter API")
    parser.add_argument("--image", "-i", required=True, help="Path to the image file")
    parser.add_argument("--model", "-m", default="anthropic/claude-3-haiku-20240307", 
                        help="Model identifier to use (default: anthropic/claude-3-haiku-20240307)")
    parser.add_argument("--prompt", "-p", default="Опиши подробно, что ты видишь на этом изображении. Ответ должен быть на русском языке.",
                        help="Text prompt to accompany the image")
    parser.add_argument("--max-tokens", "-t", type=int, default=2000,
                        help="Maximum number of tokens to generate (default: 2000)")
    parser.add_argument("--api-key", "-k", help="OpenRouter API key (overrides environment variable)")
    parser.add_argument("--output", "-o", help="Output file to save the response (JSON format)")
    parser.add_argument("--clothing-list", "-c", action="store_true", 
                        help="Generate a numbered list of clothing items seen in the image")
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not Path(args.image).exists():
        logger.error(f"Ошибка: Файл изображения не найден: {args.image}")
        return 1
    
    # Check if API key is available
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("Ошибка: API ключ OpenRouter не найден. Укажите его через аргумент --api-key или переменную окружения OPENROUTER_API_KEY")
        return 1
    
    try:
        client = OpenRouterClient(api_key=args.api_key)
        logger.info(f"Отправка изображения {args.image} в OpenRouter с моделью {args.model}")
        
        # If clothing list is requested, use a specialized prompt
        if args.clothing_list:
            prompt = """Проанализируй одежду, которую носит человек на фотографии, и создай пронумерованный список всех элементов одежды. 
            
Правила для создания списка:
1. Каждый элемент одежды должен быть указан отдельным пунктом.
2. Включи точное описание цвета и материала, если это возможно определить.
3. Опиши стиль и посадку каждого предмета.
4. Не включай предположения или неопределённые описания, если что-то не видно чётко.
5. Если виден узор или принт, опиши его.
6. Включи аксессуары и обувь, если они видны.

Формат списка должен быть таким:
1. [Предмет одежды] - [цвет], [материал], [другие важные детали]
2. [Предмет одежды] - [цвет], [материал], [другие важные детали]
И так далее.

Ответ должен быть ТОЛЬКО на русском языке и представлять собой ТОЛЬКО пронумерованный список без вступления и заключения."""
        else:
            prompt = args.prompt
        
        response = client.process_image(
            image_path=args.image,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens
        )
        
        # Pretty print the response
        print(json.dumps(response, indent=2, ensure_ascii=False))
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            logger.info(f"Response saved to {args.output}")
            
        # Extract and display the model's response text
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            if args.clothing_list:
                print("\nСписок одежды:")
            else:
                print("\nОтвет модели:")
            print(content)
        else:
            print("\nПредупреждение: В ответе нет содержимого в ожидаемом формате.")
            print("Полный ответ API выше.")
        
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 