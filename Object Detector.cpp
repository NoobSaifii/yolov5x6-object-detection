#include <fstream>
#include <opencv2/opencv.hpp>
#include <atomic>
#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <sstream>
#include <mutex>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace chrono;

// Load class names from file
vector<string> load_class_list() {
    vector<string> class_list;
    ifstream ifs("classes.txt");
    string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

// Load the YOLOv5 model
void load_net(Net& net, bool is_cuda) {
    auto result = readNet("yolov5x6.onnx");
    if (is_cuda) {
        cout << "Attempting to use CUDA\n";
        result.setPreferableBackend(DNN_BACKEND_CUDA);
        result.setPreferableTarget(DNN_TARGET_CUDA_FP16);
    }
    else {
        cout << "Running on CPU\n";
        result.setPreferableBackend(DNN_BACKEND_OPENCV);
        result.setPreferableTarget(DNN_TARGET_CPU);
    }
    net = result;
}

const vector<Scalar> colors = { Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0) };
const float INPUT_WIDTH = 640.0f;
const float INPUT_HEIGHT = 640.0f;
const float SCORE_THRESHOLD = 0.2f;
const float NMS_THRESHOLD = 0.4f;
const float CONFIDENCE_THRESHOLD = 0.4f;

struct Detection {
    int class_id;
    float confidence;
    Rect box;
};

// Detect objects in the frame - FIXED FUNCTION
void detect(Mat& image, Net& net, vector<Detection>& output, const vector<string>& className) {
    // --- 1. PREPROCESSING ---
    // Proper letterboxing to resize image without distortion
    float scale = min(INPUT_WIDTH / (float)image.cols, INPUT_HEIGHT / (float)image.rows);
    int scaled_w = (int)(image.cols * scale);
    int scaled_h = (int)(image.rows * scale);
    int pad_x = (INPUT_WIDTH - scaled_w) / 2;
    int pad_y = (INPUT_HEIGHT - scaled_h) / 2;

    Mat blob_image = Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3);
    Mat resized_image;
    resize(image, resized_image, Size(scaled_w, scaled_h));
    resized_image.copyTo(blob_image(Rect(pad_x, pad_y, scaled_w, scaled_h)));

    Mat blob;
    // Create blob. `swapRB=true` handles BGR to RGB conversion. No extra cvtColor needed.
    blobFromImage(blob_image, blob, 1.0 / 255.0, Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    // --- 2. FORWARD PASS ---
    net.setInput(blob);
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // --- 3. POSTPROCESSING ---
    float* data = (float*)outputs[0].data;
    const int rows = outputs[0].size[1];
    const int dimensions = outputs[0].size[2]; // Should be 85 (x,y,w,h,conf + 80 classes)

    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float* classes_scores = data + 5;
            Mat scores(1, className.size(), CV_32FC1, classes_scores);
            Point class_id_point;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

            if (max_class_score > SCORE_THRESHOLD) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id_point.x);

                float cx = data[0];
                float cy = data[1];
                float w = data[2];
                float h = data[3];

                // Scale box coordinates from 640x640 to original image size
                int left = (int)((cx - w / 2 - pad_x) / scale);
                int top = (int)((cy - h / 2 - pad_y) / scale);
                int width = (int)(w / scale);
                int height = (int)(h / scale);

                boxes.push_back(Rect(left, top, width, height));
            }
        }
        data += dimensions;
    }

    // --- 4. NMS ---
    vector<int> nms_result;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    output.clear(); // Clear old results
    for (int idx : nms_result) {
        output.push_back({ class_ids[idx], confidences[idx], boxes[idx] });
    }
}


// Process each camera feed and detect Objects
void process_camera(int camera_id, Net& net, const vector<string>& class_list, atomic<int>& vehicle_count, mutex& count_mutex) {
    VideoCapture capture(camera_id);
    if (!capture.isOpened()) {
        cerr << "Error opening camera: " << camera_id << endl;
        return;
    }

    int frame_skip = 10;
    int frame_counter = 0;
    while (true) {
        Mat frame;
        capture.read(frame);
        if (frame.empty()) {
            cout << "End of stream\n";
            break;
        }

        if (++frame_counter % frame_skip != 0) {
            continue;
        }

        vector<Detection> output;
        detect(frame, net, output, class_list);

        int detected_Objects = output.size();
        {
            lock_guard<mutex> lock(count_mutex);
            vehicle_count += detected_Objects;
        }

        for (int i = 0; i < detected_Objects; ++i) {
            auto detection = output[i];
            auto box = detection.box;
            auto classId = detection.class_id;
            const auto color = colors[classId % colors.size()];
            rectangle(frame, box, color, 1);
            rectangle(frame, Point(box.x, box.y - 20), Point(box.x + box.width, box.y), color, FILLED);
            putText(frame, class_list[classId].c_str(), Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
        }

        ostringstream detections_label;
        detections_label << "Active Objects: " << detected_Objects;
        string detections_label_str = detections_label.str();
        putText(frame, detections_label_str.c_str(), Point(10, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        imshow("Camera " + to_string(camera_id), frame);

        if (waitKey(1) != -1) {
            capture.release();
            cout << "finished by user\n";
            break;
        }
    }
}

int main(int argc, char** argv) {
    vector<string> class_list = load_class_list();
    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
    Net net;
    load_net(net, is_cuda);

    atomic<int> vehicle_count(0);
    mutex count_mutex;

    thread camera1(process_camera, 0, ref(net), ref(class_list), ref(vehicle_count), ref(count_mutex));
    thread camera2(process_camera, 1, ref(net), ref(class_list), ref(vehicle_count), ref(count_mutex));
    thread camera3(process_camera, 2, ref(net), ref(class_list), ref(vehicle_count), ref(count_mutex));
    thread camera4(process_camera, 3, ref(net), ref(class_list), ref(vehicle_count), ref(count_mutex));

    camera1.join();
    camera2.join();
    camera3.join();
    camera4.join();

    cout << "Total Objects detected across all cameras: " << vehicle_count.load() << endl;

    return 0;
}