#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Global variables to store the model and processor
MODEL = None
PROCESSOR = None

def initialize_model():
    """
    Initialize the model and processor if they haven't been loaded yet.
    This ensures they are only loaded once.
    """
    global MODEL, PROCESSOR
    
    if MODEL is None or PROCESSOR is None:
        print("Loading model and processor...")
        MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
        )
        PROCESSOR = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        print("Model and processor loaded successfully.")


def analyze_image_file(image_path, prompt="Опиши одежду человека на фотографии.", max_tokens=128):
    """
    Analyze an image file and return the generated description.
    
    Args:
        image_path (str): Path to the image file
        prompt (str): Text prompt to accompany the image
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: The generated description
    """
    # Initialize model if not already loaded
    initialize_model()
    
    # Check if the image file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Create message with image and prompt
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Prepare inputs for the model
    text = PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = PROCESSOR(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(MODEL.device)
    
    # Generate output
    with torch.no_grad():  # Disable gradient calculation for inference
        generated_ids = MODEL.generate(**inputs, max_new_tokens=max_tokens)
        
    # Process the generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = PROCESSOR.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0] if output_text else ""


# Example usage in a notebook or script
if __name__ == "__main__":
    # This will only run when the script is executed directly, not when imported
    import sys
    
    # Simple command-line handling that works in both regular Python and Jupyter/Colab
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        image_path = sys.argv[1]
        result = analyze_image_file(image_path)
        print(result)
    else:
        print("Usage: python visual_model.py <image_path>")
        print("Or import and use the analyze_image_file function directly.")
