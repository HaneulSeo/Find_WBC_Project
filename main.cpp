#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace cv;

// ================= CONFIG =================
const string MODEL_PATH = "rt_detr.torchscript"; 
const string IMAGE_PATH = "img_294.png";

const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;

const float SCORE_THRESHOLD = 0.30; 
const float NMS_THRESHOLD = 0.45;   
// ==========================================

Mat preprocess_letterbox(const Mat& img, int target_w, int target_h, float& scale, int& x_offset, int& y_offset) {
    int w = img.cols;
    int h = img.rows;
    scale = min((float)target_w / w, (float)target_h / h);
    int new_w = round(w * scale);
    int new_h = round(h * scale);
    Mat resized;
    resize(img, resized, Size(new_w, new_h));
    Mat canvas = Mat::ones(target_h, target_w, CV_8UC3) * 114;
    x_offset = (target_w - new_w) / 2;
    y_offset = (target_h - new_h) / 2;
    resized.copyTo(canvas(Rect(x_offset, y_offset, new_w, new_h)));
    return canvas;
}

int main() {
    // [1] 디바이스 설정
    torch::Device device(torch::kCPU);
    if (torch::hasMPS()) {
        cout << ">>> GPU 가속" << endl;
        device = torch::Device(torch::kMPS);
    } else {
        cout << ">>> CPU를 사용합니다." << endl;
    }

    // [2] 모델 로드
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(MODEL_PATH);
        module.to(device);
        
        // 모델을 Half(반정밀도)로 변환
        if (torch::hasMPS()) {
            cout << ">>> FP16  적용" << endl;
            module.to(torch::kHalf);
        }
        
        module.eval();
    } catch (const c10::Error& e) {
        cerr << "모델 로드 실패: " << e.msg() << endl;
        return -1;
    }

    // [3] 이미지 로드 및 전처리
    Mat img = imread(IMAGE_PATH);
    if (img.empty()) { cerr << "이미지 없음" << endl; return -1; }

    float scale;
    int x_offset, y_offset;
    Mat input_img = preprocess_letterbox(img, INPUT_WIDTH, INPUT_HEIGHT, scale, x_offset, y_offset);

    Mat rgb_img;
    cvtColor(input_img, rgb_img, COLOR_BGR2RGB);
    rgb_img.convertTo(rgb_img, CV_32F, 1.0 / 255.0);
    
    torch::Tensor input_tensor = torch::from_blob(rgb_img.data, {INPUT_HEIGHT, INPUT_WIDTH, 3});
    
    input_tensor = input_tensor.permute({2, 0, 1}).unsqueeze(0).to(device);
    if (torch::hasMPS()) {
        input_tensor = input_tensor.to(torch::kHalf);
    }

    cout << ">>> 벤치마크 시작 (50회 반복)..." << endl;
    
    // 워밍업
    module.forward({input_tensor}); 

    double total_time = 0.0;
    torch::Tensor output; 

    int TEST_COUNT = 50;
    for (int i = 0; i < TEST_COUNT; i++) {
        auto t_start = chrono::high_resolution_clock::now();

        // 추론
        torch::IValue output_ivalue = module.forward({input_tensor});
        
        if (output_ivalue.isTuple()) {
            output = output_ivalue.toTuple()->elements()[0].toTensor();
        } else {
            output = output_ivalue.toTensor();
        }
        
        output = output.to(torch::kFloat).to(torch::kCPU);

        auto t_end = chrono::high_resolution_clock::now();
        chrono::duration<double, std::milli> ms = t_end - t_start;
        total_time += ms.count();
        
        if (i % 10 == 0) cout << "."; 
    }
    cout << endl;

    // [5] 결과 파싱
    if (output.size(1) < output.size(2)) {
        output = output.transpose(1, 2);
    }
    
    torch::Tensor result_tensor = output[0];
    // float로 변환했으므로 accessor<float> 사용 가능
    auto data = result_tensor.accessor<float, 2>();
    
    int num_rows = output.size(1);
    int num_cols = output.size(2);

    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;



    for (int i = 0; i < num_rows; i++) {
        float max_score = -1.0f;
        int class_id = -1;
        
        for (int c = 4; c < num_cols; c++) {
            if (data[i][c] > max_score) {
                max_score = data[i][c];
                class_id = c - 4;
            }
        }

        if (max_score > SCORE_THRESHOLD) {
            float cx = data[i][0];
            float cy = data[i][1];
            float w  = data[i][2];
            float h  = data[i][3];

            // 0~1 정규화된 좌표 복원
            if (cx < 1.0 && w < 1.0) { 
                cx *= INPUT_WIDTH;
                cy *= INPUT_HEIGHT;
                w  *= INPUT_WIDTH;
                h  *= INPUT_HEIGHT;
            }
            
            int left = (int)((cx - 0.5 * w - x_offset) / scale);
            int top  = (int)((cy - 0.5 * h - y_offset) / scale);
            int width = (int)(w / scale);
            int height = (int)(h / scale);

            boxes.push_back(Rect(left, top, width, height));
            confidences.push_back(max_score);
            class_ids.push_back(class_id);
        }
    }

    // [6] NMS
    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    cout << "------------------------------------------" << endl;
    cout << " [ LibTorch (FP16) 최종 결과 ]" << endl;
    cout << " 평균 속도: " << total_time / TEST_COUNT << " ms" << endl;
    cout << " 발견된 개수: " << indices.size() << "개" << endl;
    if (indices.size() == 17) cout << " ★ 성공! (17개) ★" << endl;
    cout << "------------------------------------------" << endl;

    for (int idx : indices) {
        rectangle(img, boxes[idx], Scalar(0, 255, 0), 2);
        string label = format("%.0f%%", confidences[idx] * 100);
        putText(img, label, Point(boxes[idx].x, boxes[idx].y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    }
    imwrite("result_libtorch_fp16.jpg", img);

    return 0;
}