import os
import glob
import pandas as pd
from ultralytics import YOLO

SOURCE_PATH = 'img_294.png' 

def run_benchmark():
    # 1. 모델 파일 확인
    if not os.path.exists("./models"):
        print("'models' 폴더가 없습니다. 폴더를 만들고 .pt 파일을 넣어주세요.")
        return

    model_files = glob.glob("./models/*.pt")
    model_files.sort()
    
    if not model_files:
        print("'./models' 폴더에 .pt 파일이 없습니다.")
        return

    benchmark_data = []

    print(f"\n[벤치마크 시작] 발견된 모델: {len(model_files)}개")
    print(f"테스트 이미지: {SOURCE_PATH}")

    for file in model_files:
        base_name = os.path.splitext(file)[0]      # 예: ./models/yolo10s
        model_name = os.path.basename(base_name)   # 예: yolo10s
        engine_file = f"{base_name}.engine"        # 예: ./models/yolo10s.engine
        
        # ---------------------------------------------------------------------
        # RT-DETR: 구조적 호환성 문제로 FP16 변환 시 정확도 급락 -> FP32 사용
        # YOLO: FP16 변환 시 속도 대폭 향상 및 정확도 유지 -> FP16 사용
        # ---------------------------------------------------------------------
        if "rt_detr" in model_name.lower():
            use_half = False
            mode_desc = "FP32 (정확도 보존)"
        else:
            use_half = True
            mode_desc = "FP16 (속도 최적화)"

        print(f"👉 [{model_name}] 작업 시작 ({mode_desc})...")

        try:
            if os.path.exists(engine_file):
                print(f"기존 엔진(.engine) 발견 변환 건너뜀.")
                model = YOLO(engine_file)
            else:
                print(f"엔진 파일 없음. TensorRT 변환 시작")
                pt_model = YOLO(file)
                # device=0 (GPU 0번) 필수
                exported_path = pt_model.export(format='engine', device=0, half=use_half, verbose=False)
                model = YOLO(exported_path)
            
            print("GPU 예열", end=" ")
            for _ in range(10):
                model.predict(source=SOURCE_PATH, verbose=False, device=0)
            print("완료.")

            # 4. Benchmark (성능 측정 50회)
            print("측정 진행 중...", end=" ")
            inference_times = []
            detected_counts = []
            
            for _ in range(50):
                result = model.predict(source=SOURCE_PATH, save=False, verbose=False, device=0)
                inference_times.append(result[0].speed['inference'])
                detected_counts.append(len(result[0].boxes))
            
            # 평균 계산
            avg_infer = sum(inference_times) / len(inference_times)
            avg_count = sum(detected_counts) / len(detected_counts)
            fps = 1000 / avg_infer if avg_infer > 0 else 0
            
            benchmark_data.append({
                "Model": model_name,
                "Type": "TensorRT",
                "Precision": "FP16" if use_half else "FP32",
                "Avg Objects": round(avg_count, 1),
                "Inference (ms)": round(avg_infer, 2),
                "FPS": round(fps, 1)
            })
            print(f"성공! (FPS: {fps:.1f})")
            print("-" * 60)

        except Exception as e:
            print(f"\n실패: {e}")
            print("-" * 60)

    # 5. 최종 결과표 출력
    if benchmark_data:
        df = pd.DataFrame(benchmark_data)
        df = df.sort_values(by="FPS", ascending=False)
        
        print("\n" + "="*75)
        print("YOLO & RT-DETR TensorRT 벤치마크 최종 결과 🏆")
        print("="*75)
        
        try:
            from tabulate import tabulate
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        except ImportError:
            print(df.to_string(index=False))
    else:
        print("결과 데이터가 없습니다.")

if __name__ == "__main__":
    run_benchmark()