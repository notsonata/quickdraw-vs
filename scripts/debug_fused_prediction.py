#!/usr/bin/env python3
import sys
import os
import json
import argparse
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from draw_game import preprocess, classifier
from draw_game.config import settings

def _print_tensor_info(name: str, tensor: np.ndarray):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype.name}, min={np.min(tensor):.4f}, max={np.max(tensor):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Debug script for fused TFLite model predictions")
    parser.add_argument("--image", type=str, help="Path to raw image file (PNG/JPG)")
    parser.add_argument("--strokes", type=str, help="Path to stroke events JSON file")
    args = parser.parse_args()

    # Determine real image tensor
    if args.image and os.path.exists(args.image):
        print(f"Loading image from {args.image}...")
        raw_image = cv2.imread(args.image)
        if raw_image is None:
            print("Failed to load image, using blank.")
            raw_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    else:
        print("No image provided (or file not found), using dummy raw image (blank white canvas).")
        raw_image = np.ones((600, 800, 3), dtype=np.uint8) * 255

    real_image_tensor = preprocess.preprocess_image_for_fused(raw_image, size=settings.MODEL_IMAGE_SIZE)

    # Determine real stroke tensor
    if args.strokes and os.path.exists(args.strokes):
        print(f"Loading strokes from {args.strokes}...")
        with open(args.strokes, "r") as f:
            stroke_events = json.load(f)
        real_stroke_tensor = preprocess.preprocess_strokes(stroke_events)
        if real_stroke_tensor is None:
            print("Not enough strokes in file, using dummy zero tensor.")
            real_stroke_tensor = np.zeros((1, settings.MODEL_SEQ_LEN, settings.MODEL_FEATURES), dtype=np.float32)
            real_stroke_tensor[0, 0, 4] = 1.0  # End token
    else:
        print("No stroke file provided, using dummy zero tensor (with end token).")
        real_stroke_tensor = np.zeros((1, settings.MODEL_SEQ_LEN, settings.MODEL_FEATURES), dtype=np.float32)
        real_stroke_tensor[0, 0, 4] = 1.0

    print("\n--- Model Inputs ---")
    _print_tensor_info("Real Image Tensor", real_image_tensor)
    _print_tensor_info("Real Stroke Tensor", real_stroke_tensor)

    # Create model
    print("\nLoading Classifier...")
    clf = classifier.create_classifier()
    
    print("\n--- Model Details ---")
    if hasattr(clf, "input_details"):
        for i, input_detail in enumerate(clf.input_details):
            print(f"Input {i}: name={input_detail['name']}, shape={input_detail['shape']}, dtype={input_detail['dtype']}")
        for i, output_detail in enumerate(clf.output_details):
            print(f"Output {i}: name={output_detail['name']}, shape={output_detail['shape']}, dtype={output_detail['dtype']}")
    else:
        print("Model does not expose input_details directly (likely a StubClassifier).")

    # 1. fused_normal
    print("\n=== Mode 1: fused_normal ===")
    res_fused = clf.predict({"image": real_image_tensor, "stroke": real_stroke_tensor})
    print("Top 5 Predictions:")
    for label, conf in res_fused["top5"]:
        print(f"  {label}: {conf:.4f}")

    # 2. image_only
    print("\n=== Mode 2: image_only ===")
    zero_stroke_tensor = np.zeros((1, settings.MODEL_SEQ_LEN, settings.MODEL_FEATURES), dtype=np.float32)
    zero_stroke_tensor[0, 0, 4] = 1.0
    res_image = clf.predict({"image": real_image_tensor, "stroke": zero_stroke_tensor})
    print("Top 5 Predictions:")
    for label, conf in res_image["top5"]:
        print(f"  {label}: {conf:.4f}")

    # 3. stroke_only
    print("\n=== Mode 3: stroke_only ===")
    blank_image_tensor = np.ones((1, settings.MODEL_IMAGE_SIZE, settings.MODEL_IMAGE_SIZE, 1), dtype=np.float32)
    res_stroke = clf.predict({"image": blank_image_tensor, "stroke": real_stroke_tensor})
    print("Top 5 Predictions:")
    for label, conf in res_stroke["top5"]:
        print(f"  {label}: {conf:.4f}")


if __name__ == "__main__":
    main()
