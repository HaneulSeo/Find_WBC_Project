# 🚀 YOLO & RT-DETR TensorRT Benchmark

이 프로젝트는 PyTorch(`.pt`) 모델을 **NVIDIA TensorRT(`.engine`)**로 자동 변환하고, 추론 속도(FPS)와 정확도를 비교 분석하는 벤치마크 도구입니다.

YOLO 시리즈와 RT-DETR 모델의 **TensorRT 최적화 성능**을 즉시 확인할 수 있습니다.

## ✨ 주요 특징 (Key Features)
- **자동 변환 (Auto Export):** `.pt` 파일만 넣으면 자동으로 `.engine` 파일 생성
- **스마트 최적화 (Smart Optimization):**
  - **YOLOv8 / v10:** `FP16` (반정밀도) 적용 → **속도 극대화**
  - **RT-DETR:** `FP32` (단정밀도) 적용 → **정확도 손실 방지**
- **결과 리포트:** 모델별 FPS, 추론 시간, 평균 탐지 개수 비교표 출력

## 🛠️ 환경 요구사항 (Requirements)
- **NVIDIA GPU** (필수)
- OS: Linux (Ubuntu 등) 또는 Windows (WSL2 권장)
- Python 3.8+
- CUDA Toolkit 설치 권장

## 📦 설치 및 실행 (Quick Start)

### 1. 설치
```bash
git clone [이 레포지토리 URL]
cd [레포지토리 이름]
pip install -r requirements.txt
python3 tensorRT_Benchmark.py