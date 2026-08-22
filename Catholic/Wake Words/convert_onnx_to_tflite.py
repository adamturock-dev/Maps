import json
import shutil
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage:")
    print("  python convert.py <model.onnx>")
    sys.exit(1)

onnx_file = Path(sys.argv[1]).resolve()

if not onnx_file.exists():
    print(f"ERROR: File not found: {onnx_file}")
    sys.exit(1)

model_name = onnx_file.stem

tflite_file = onnx_file.with_suffix(".tflite")
json_file = onnx_file.with_suffix(".json")

saved_model_dir = onnx_file.parent / f"{model_name}_saved_model"

print(f"Input : {onnx_file}")
print(f"Model : {model_name}")

# Remove previous SavedModel if it exists
if saved_model_dir.exists():
    shutil.rmtree(saved_model_dir)

# ONNX -> TensorFlow SavedModel
print("\n[1/3] Converting ONNX -> TensorFlow SavedModel...")

subprocess.run(
    [
        sys.executable,
        "-m",
        "onnx2tf",
        "-i",
        str(onnx_file),
        "-o",
        str(saved_model_dir),
    ],
    check=True,
)

# SavedModel -> TFLite
print("[2/3] Converting SavedModel -> TFLite...")

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(
    str(saved_model_dir)
)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(tflite_file, "wb") as f:
    f.write(tflite_model)

# Create microWakeWord JSON
print("[3/3] Creating JSON manifest...")

manifest = {
    "type": "micro",
    "wake_word": model_name,
    "author": "Adam Turock",
    "website": "",
    "version": 1,
    "model": tflite_file.name
}

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

# Cleanup
if saved_model_dir.exists():
    shutil.rmtree(saved_model_dir)

print("\nSUCCESS")
print(f"Created: {tflite_file}")
print(f"Created: {json_file}")