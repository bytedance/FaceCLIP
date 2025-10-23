#!/bin/bash

sudo apt-get install -y python3-opencv
pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip3 install -e .
pip3 install -e core/face_clip/GroundingDINO
pip3 install timm
pip3 install bson
pip3 install beautifulsoup4
pip3 install omegaconf
pip3 install accelerate==0.32.0
pip3 install ftfy
pip3 install insightface
pip3 install opencv-python
pip3 install diffusers==0.33.1
pip3 install transformers==4.45.2
pip3 install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip3 install httpx==0.23
pip3 install tenacity
pip3 install huggingface_hub[hf_transfer]
pip3 install open_clip_torch
pip3 install protobuf==3.20
pip3 install numpy==1.23.1