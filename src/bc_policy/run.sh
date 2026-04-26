#!/bin/bash
cd "$(dirname "$0")"
python3 inference_node.py --ros-args \
    -p model_path:=final_model.onnx \
    -p config_path:=config.yaml \
    -p use_onnx:=true \
    -p max_speed:=2.0
