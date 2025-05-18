#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to use the OpenRouterClient
to process images programmatically.
"""

from openrouter_image_client import OpenRouterClient
import os
from pathlib import Path
import logging
import json
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def main():
    # Initialize the client with API key from environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("Ошибка: API ключ OpenRouter не найден в переменных окружения")
        return 1
        
    client = OpenRouterClient(api_key=api_key)
    
    # Path to your image
    image_path = "niger.jpg"  # Replace with your image path
    
    # Check if image exists
    if not Path(image_path).exists():
        logger.error(f"Ошибка: Файл изображения не найден: {image_path}")
        logger.info("Убедитесь, что у вас есть файл изображения и укажите правильный путь")
        return 1
    
    # Example: Numbered list of clothing items using Claude model
    print("Пронумерованный список одежды на человеке (Claude)")
    try:
        # Используем модель Claude с детальным промптом для получения пронумерованного списка
        response = client.process_image(
            image_path=image_path,
            model="anthropic/claude-3-haiku-20240307",
            prompt="""Проанализируй одежду, которую носит человек на фотографии, и создай пронумерованный список всех элементов одежды. 
            
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

Ответ должен быть ТОЛЬКО на русском языке и представлять собой ТОЛЬКО пронумерованный список без вступления и заключения.""",
            max_tokens=2000
        )
        
        # Сохраняем полный ответ для отладки
        with open("debug_response.json", "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        
        # Print only the content from the response
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            print(f"\nСписок одежды:\n{content}\n")
        else:
            print("Ответ не содержит ожидаемого содержимого. Проверьте файл debug_response.json")
            print(f"Полный ответ: {json.dumps(response, ensure_ascii=False)}")
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}")
        
    # Alternative model if first one failed
    print("\nАльтернативная модель (если первая не сработала)")
    try:
        # Пробуем модель gemini для получения пронумерованного списка
        response = client.process_image(
            image_path=image_path,
            model="google/gemini-pro-vision",
            prompt="""Ты эксперт по анализу одежды. Внимательно рассмотри фотографию и составь пронумерованный список всех элементов одежды, которые видны на человеке.

Создай точный пронумерованный список в следующем формате:
1. [Название предмета] - [цвет], [фасон/стиль], [материал если видно]
2. [Название предмета] - [цвет], [фасон/стиль], [материал если видно]

Список должен включать ВСЕ видимые элементы одежды, включая верхнюю одежду, нижнюю одежду, обувь и аксессуары.
Пиши ТОЛЬКО на русском языке.
Не добавляй введение или заключение, только пронумерованный список.""",
            max_tokens=2000
        )
        
        # Print only the content from the response
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            print(f"\nСписок одежды (Gemini):\n{content}\n")
        else:
            print("Gemini не вернул ожидаемый формат ответа.")
    except Exception as e:
        logger.error(f"Ошибка при использовании Gemini: {str(e)}")

if __name__ == "__main__":
    main() 