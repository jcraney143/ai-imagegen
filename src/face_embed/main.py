import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# Initialize InsightFace
def init_model():
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

# Extract face embedding from image
def extract_embedding(image_path, output_dir):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    app = init_model()
    faces = app.get(img)

    if not faces:
        raise ValueError("No face detected in image.")

    face = faces[0]
    aligned = face_align.norm_crop(img, landmark=face.landmark_2d_106)
    embedding = app.model.get_feat(aligned).flatten()

    # Save embedding using image filename
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{image_name}_embedding.npy")
    np.save(output_path, embedding)

    return output_path

# CLI entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract face embedding from selfie")
    parser.add_argument("--image", required=True, help="Path to selfie image")
    parser.add_argument("--output_dir", default="data/output", help="Directory to save embedding")
    args = parser.parse_args()

    saved_path = extract_embedding(args.image, args.output_dir)
    print(f"Embedding saved to {saved_path}")
