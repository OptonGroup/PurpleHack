import torch
from rembg import remove
from PIL import Image
import io
device = "cuda" if torch.cuda.is_available() else "cpu"

def remove_bg(input_path, output_path):
    with open(input_path, "rb") as f:
        input_image = f.read()
    
    output_image = remove(input_image, alpha_matting=True, alpha_matting_foreground_threshold=240, 
                          alpha_matting_background_threshold=10, alpha_matting_erode_size=5)

    with open(output_path, "wb") as f:
        f.write(output_image)

remove_bg("f2a0f6e9_20ab_11ee_a5c0_d8d385af0808.webp", "output.png")
