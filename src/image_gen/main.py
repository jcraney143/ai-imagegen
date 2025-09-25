import os
import torch
import numpy as np
from diffusers import StableDiffusionPipeline, IPAdapter
from PIL import Image

# Load embeddings
def load_embeddings(embedding_dir):
    selfie_path = os.path.join(embedding_dir, "selfie_embedding.npy")
    prompt_path = os.path.join(embedding_dir, "prompt_embedding.npy")

    if not os.path.exists(selfie_path) or not os.path.exists(prompt_path):
        raise FileNotFoundError("Missing embedding files in output directory.")

    selfie_embedding = np.load(selfie_path)
    prompt_embedding = np.load(prompt_path)

    return selfie_embedding, prompt_embedding

# Initialize pipeline
def init_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    ip_adapter = IPAdapter(pipe)
    return pipe, ip_adapter

# Generate image
def generate_image(embedding_dir, output_dir):
    selfie_embedding, prompt_embedding = load_embeddings(embedding_dir)
    pipe, ip_adapter = init_pipeline()

    # Apply IP-Adapter conditioning
    image = ip_adapter.generate(
        prompt_embedding=torch.tensor(prompt_embedding).to("cuda"),
        identity_embedding=torch.tensor(selfie_embedding).to("cuda"),
        guidance_scale=7.5,
        num_inference_steps=30
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "result.png")
    image.save(output_path)

    return output_path

# CLI entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate image from embeddings")
    parser.add_argument("--embedding_dir", default="data/output", help="Directory with .npy embeddings")
    parser.add_argument("--output_dir", default="data/output", help="Directory to save generated image")
    args = parser.parse_args()

    result_path = generate_image(args.embedding_dir, args.output_dir)
    print(f"Image saved to {result_path}")
