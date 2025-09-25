import os
import sys
import logging
from datetime import datetime
import subprocess

# Setup logging
log_path = "data/logs/pipeline.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_stage(description, command):
    logging.info(f"Starting: {description}")
    print(f"🔧 {description}...")
    try:
        subprocess.run(command, shell=True, check=True)
        logging.info(f"Completed: {description}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed: {description} — {e}")
        print(f"❌ Error in {description}")
        sys.exit(1)

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Pipeline started at {timestamp}")

    # Stage 1: Face Embedding
    run_stage("Face Embedding", "python src/face_embed/main.py --image data/input/selfie.jpg --output_dir data/output")

    # Stage 2: Prompt Encoding
    run_stage("Prompt Encoding", "python src/prompt_encoder/main.py --prompt data/input/prompt.json --output_dir data/output")

    # Stage 3: Image Generation
    run_stage("Image Generation", "python src/image_gen/main.py --embedding_dir data/output --output_dir data/output")

    # Stage 4: Post-Processing
    run_stage("Post-Processing", "python src/postfx/main.py --input data/output/result.png --output data/output/result_upscaled.png")

    logging.info("Pipeline completed successfully.")
    print("✅ Image generation pipeline completed.")

if __name__ == "__main__":
    main()
