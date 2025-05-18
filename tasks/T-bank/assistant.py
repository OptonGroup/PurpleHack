from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain.chat_models.gigachat import GigaChat
# from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
import os
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

roles = {
    "Ассистент-стилист": "Ты ассистент-стилист. Твоя задача — собрать капсульный гардероб или подобрать несколько луков (с чем носить элемент одежды)",
    "Ассистент-косметолог": "Ты ассистент-косметолог. Твоя задача — подобрать уход под тип кожи и образ жизни пользователя",
    "Ассистент-нутрициолог": "Ты ассистент-нутрициолог. Твоя задача — подобрать продуктовую корзину под КБЖУ и бюджет пользователя.",
    "Ассистент-дизайнер": "Ты ассистент-дизайнер. Твоя задача — подобрать сезонный декор для дома или описать, как лучше обставить комнату от мебели до мелких деталей. Пиши максимально подробно."
}

class ConversationMemory:
    def __init__(self):
        self.conversations = {}
    
    def add_message(self, user_id: str, role: str, content: str):
        """Add a message to the conversation history for a specific user"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({"role": role, "content": content})
    
    def get_history(self, user_id: str) -> List[Dict[str, str]]:
        """Get the conversation history for a specific user"""
        return self.conversations.get(user_id, [])
    
    def clear_history(self, user_id: str):
        """Clear the conversation history for a specific user"""
        if user_id in self.conversations:
            self.conversations[user_id] = []


class ChatAssistant:
    def __init__(
        self,
        model_type: str = "gigachat",  # Options: "gigachat", "huggingface", "openrouter"
        model_name: Optional[str] = None,
        gigachat_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        default_role: str = "generic",
        max_retries: int = 3,
        timeout: int = 30,
        fallback_to_local: bool = True
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.gigachat_api_key = gigachat_api_key
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_base_url = openrouter_base_url
        self.default_role = default_role
        self.max_retries = max_retries
        self.timeout = timeout
        self.fallback_to_local = fallback_to_local
        
        # Initialize conversation memory
        self.memory = ConversationMemory()
        
        # Initialize LLM based on model_type
        try:
            self.llm = self._initialize_llm()
            logger.info(f"Successfully initialized {model_type} model")
        except Exception as e:
            logger.error(f"Error initializing {model_type} model: {str(e)}")
            if fallback_to_local and model_type != "huggingface":
                logger.info("Falling back to local model")
                self.model_type = "huggingface"
                self.model_name = "google/gemma-2b-it"  # Use a smaller model as fallback
                try:
                    self.llm = self._initialize_llm()
                    logger.info(f"Successfully initialized fallback model")
                except Exception as e:
                    logger.error(f"Error initializing fallback model: {str(e)}")
                    self.llm = None
            else:
                self.llm = None
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on model_type"""
        if self.model_type == "gigachat":
            if not self.gigachat_api_key:
                raise ValueError("GigaChat API key is required for GigaChat model")
            return GigaChat(
                credentials=self.gigachat_api_key,
                verify_ssl_certs=False
            )
        
        elif self.model_type == "huggingface":
            if not self.model_name:
                raise ValueError("Model name is required for HuggingFace model")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(self.model_name)
                pipe = pipeline(
                    'text-generation', 
                    model=model, 
                    tokenizer=tokenizer, 
                    max_new_tokens=512
                )
                return HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                logger.error(f"Error loading HuggingFace model: {str(e)}")
                logger.info("Attempting to use tokenizer-only mode")
                # If full model loading fails, try to use just the tokenizer for text generation
                # This is a simplified fallback that won't work for actual inference
                # but prevents the application from crashing
                return None
        
        elif self.model_type == "openrouter":
            if not self.openrouter_api_key:
                raise ValueError("OpenRouter API key is required for OpenRouter model")
            # OpenRouter client will be used directly in generate_response
            return None
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def change_model(
        self, 
        model_type: str,
        model_name: Optional[str] = None,
        gigachat_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None
    ):
        """Change the current LLM model"""
        self.model_type = model_type
        
        if model_name:
            self.model_name = model_name
        
        if gigachat_api_key:
            self.gigachat_api_key = gigachat_api_key
            
        if openrouter_api_key:
            self.openrouter_api_key = openrouter_api_key
        
        try:    
            self.llm = self._initialize_llm()
            return f"Model successfully changed to {model_type}" + (f" ({model_name})" if model_name else "")
        except Exception as e:
            error_msg = f"Failed to change model: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _format_messages_for_openrouter(self, system_prompt: str, user_input: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for OpenRouter API"""
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for message in history:
            messages.append(message)
            
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _generate_with_openrouter(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using OpenRouter API with retry logic"""
        client = OpenAI(
            base_url=self.openrouter_base_url,
            api_key=self.openrouter_api_key,
            timeout=self.timeout
        )
        
        model_to_use = self.model_name or "mistralai/mistral-7b-instruct:free"
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempt {attempt+1}/{self.max_retries} to call OpenRouter API")
                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://your-app-url.com",  # Replace with your app URL
                        "X-Title": "AI Assistant",  # Replace with your app name
                    },
                    model=model_to_use,
                    messages=messages
                )
                
                return completion.choices[0].message.content
            
            except Exception as e:
                logger.error(f"Error calling OpenRouter API (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff: wait 2^attempt seconds before retrying
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    # If all retries fail and fallback is enabled, try using local model
                    if self.fallback_to_local:
                        logger.info("All OpenRouter API attempts failed. Falling back to local response generation.")
                        return self._generate_fallback_response(messages)
                    else:
                        return f"Error: Failed to generate response after {self.max_retries} attempts. Error: {str(e)}"
    
    def _generate_fallback_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a fallback response when API calls fail"""
        # Extract the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break
        
        # Try to use a local model if available
        if self.fallback_to_local and self.model_type != "huggingface":
            original_model_type = self.model_type
            original_model_name = self.model_name
            
            try:
                self.model_type = "huggingface"
                self.model_name = "google/gemma-2b-it"  # Use a smaller model as fallback
                self.llm = self._initialize_llm()
                
                if self.llm:
                    # Format prompt for the local model
                    prompt = f"USER: {user_message}\nASSISTANT:"
                    response = self._generate_with_langchain(prompt)
                    
                    # Restore original model settings
                    self.model_type = original_model_type
                    self.model_name = original_model_name
                    self.llm = self._initialize_llm()
                    
                    return response
            except Exception as e:
                logger.error(f"Error using fallback model: {str(e)}")
                # Restore original model settings
                self.model_type = original_model_type
                self.model_name = original_model_name
                
        # If all else fails, return a generic response
        return "I'm sorry, I'm having trouble connecting to my knowledge source right now. Please try again later or ask a different question."
    
    def _generate_with_langchain(self, prompt: str) -> str:
        """Generate a response using LangChain-compatible models (GigaChat, HuggingFace)"""
        if not self.llm:
            return "Model is not properly initialized. Please check your configuration."
        
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Error generating response with LangChain: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def generate_response(self, user_id: str, role: str, user_input: str) -> str:
        """Generate a response based on user input, role, and conversation history"""
        if not user_input.strip():
            return "Please provide a non-empty message."
            
        system_prompt = roles.get(role, roles.get(self.default_role, "You are a helpful assistant."))
        
        # Get conversation history for this user
        history = self.memory.get_history(user_id)
        
        try:
            # Generate response based on model type
            if self.model_type == "openrouter":
                messages = self._format_messages_for_openrouter(system_prompt, user_input, history)
                response = self._generate_with_openrouter(messages)
            else:
                # Format prompt for GigaChat or HuggingFace
                prompt = f"SYSTEM: {system_prompt}\n\n"
                
                # Add conversation history
                for msg in history:
                    if msg["role"] == "user":
                        prompt += f"USER: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        prompt += f"ASSISTANT: {msg['content']}\n"
                
                # Add current user input
                prompt += f"USER: {user_input}\nASSISTANT:"
                
                response = self._generate_with_langchain(prompt)
            
            # Save the conversation
            self.memory.add_message(user_id, "user", user_input)
            self.memory.add_message(user_id, "assistant", response)
            
            return response
            
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            logger.error(error_message)
            return error_message
    
    def clear_conversation(self, user_id: str):
        """Clear the conversation history for a specific user"""
        self.memory.clear_history(user_id)
        return f"Conversation history cleared for user {user_id}"

assistant = ChatAssistant(openrouter_api_key='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', model_type="openrouter",
    model_name="mistralai/mistral-7b-instruct:free")

response = assistant.generate_response(
    user_id="user123",
    role="Ассистент-косметолог",
    user_input="У меня появилась сыпь на жопе. Что мне делать?"
)

response2 = assistant.generate_response(
    user_id="user123",
    role="Ассистент-косметолог",
    user_input="Что я спросил?"
)

print(response)
print(response2)


# Example of usage:
# assistant = ChatAssistant(
#     model_type="gigachat",
#     gigachat_api_key="your-gigachat-api-key"
# )
# 
# # Get response from the assistant
# response = assistant.generate_response(
#     user_id="user123",
#     role="nutritionist",
#     user_input="I want to lose weight, what should I eat?"
# )
# 
# # Continue the conversation
# response2 = assistant.generate_response(
#     user_id="user123", 
#     role="nutritionist",
#     user_input="How many calories should I consume daily?"
# )
# 
# # Change to OpenRouter model
# assistant.change_model(
#     model_type="openrouter",
#     model_name="mistralai/mistral-7b-instruct:free",
#     openrouter_api_key="your-openrouter-api-key"
# )
