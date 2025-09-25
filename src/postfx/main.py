import os
import cv2
import torch
import numpy as np
from PIL import Image
from gfpgan import GFPGANer
from realesrgan import RealESRGANer

# Paths
DEFAULT_INPUT = "data/output/result.png"
DEFAULT_OUTPUT = "data/output/result_upscaled.png"
MODEL_DIR = "models"

# Restore face with GFPGAN
def restore_face(image_path):
    gfpgan = GFPGANer(
        model_path=os.path.join(MODEL_DIR, "gfpgan", "GFPGANv1.4.pth"),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None
    )
    img = cv2.imread(image_path)
    _, _, restored_img = gfpgan.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    return restored_img

# Upscale with Real-ESRGAN
def upscale_image(image_np):
    model_path = os.path.join(MODEL_DIR, "realesrgan", "RealESRGAN_x4plus.pth")
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model_name="RealESRGAN_x4plus",
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True
    )
    output, _ = upsampler.enhance(image_np, outscale=4)
    return output

# Full pipeline
def postprocess(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT):
    restored = restore_face(input_path)
    upscaled = upscale_image(restored)
    Image.fromarray(upscaled).save(output_path)
    return output_path

# CLI entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Post-process generated image")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to generated image")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to save enhanced image")
    args = parser.parse_args()

    final_path = postprocess(args.input, args.output)
    print(f"Enhanced image saved to {final_path}")
