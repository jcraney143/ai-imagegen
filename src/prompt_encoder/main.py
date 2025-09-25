import os
import json
import torch
from transformers import Blip2Processor, Blip2Model

# Load BLIP-2 model and processor
def init_blip2():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto", torch_dtype=torch.float16)
    return processor, model

# Encode prompt into semantic embedding
def encode_prompt(prompt_path, output_dir):
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r") as f:
        prompt_data = json.load(f)

    prompt_text = prompt_data.get("prompt", "")
    if not prompt_text:
        raise ValueError("Prompt text is empty.")

    processor, model = init_blip2()
    inputs = processor(prompt=prompt_text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.detach().cpu().numpy()

    # Save embedding
    prompt_name = os.path.splitext(os.path.basename(prompt_path))[0]
    output_path = os.path.join(output_dir, f"{prompt_name}_embedding.npy")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        np.save(f, embedding)

    return output_path

# CLI entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Encode prompt using BLIP-2")
    parser.add_argument("--prompt", required=True, help="Path to prompt.json")
    parser.add_argument("--output_dir", default="data/output", help="Directory to save embedding")
    args = parser.parse_args()

    saved_path = encode_prompt(args.prompt, args.output_dir)
    print(f"Prompt embedding saved to {saved_path}")
