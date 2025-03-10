#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_model_and_processor():
    """
    Load the Qwen2.5-VL model and its processor.
    
    Returns:
        tuple: (model, processor) - The loaded model and processor
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    
    # For better performance with multiple images, uncomment the following:
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    return model, processor


def analyze_image(model, processor, image_path, prompt="Опиши одежду человека на фотографии в подробностях.", max_tokens=128):
    """
    Analyze an image with the given prompt using the Qwen2.5-VL model.
    
    Args:
        model: The loaded Qwen2.5-VL model
        processor: The model processor
        image_path (str): Path to the image file
        prompt (str): Text prompt to accompany the image
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: The generated description
    """
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
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Generate output
    with torch.no_grad():  # Disable gradient calculation for inference
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        
    # Process the generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0] if output_text else ""


def main():
    """Main function to parse arguments and run the image analysis."""
    parser = argparse.ArgumentParser(description="Analyze images using Qwen2.5-VL model")
    parser.add_argument("image_path", type=str, help="Path to the JPEG image file")
    parser.add_argument("--prompt", type=str, 
                      default="Опиши одежду человека на фотографии в подробностях.",
                      help="Text prompt to use with the image")
    parser.add_argument("--max-tokens", type=int, default=128,
                      help="Maximum number of tokens to generate")
    
    args = parser.parse_args()
    
    try:
        # Load model and processor
        model, processor = load_model_and_processor()
        
        # Analyze the image
        result = analyze_image(
            model, 
            processor, 
            args.image_path, 
            prompt=args.prompt,
            max_tokens=args.max_tokens
        )
        
        # Print the result
        print(result)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())