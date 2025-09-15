# Vehicle Detection with YOLOv5 🚗🚌🚛

A **C++ project** using **OpenCV DNN** and **YOLOv5 (ONNX format)** to detect and count vehicles from multiple camera feeds in real time.

---

## 📌 Features

* Loads **YOLOv5n (nano)** model in ONNX format for lightweight real-time inference.
* Runs detection on **multiple cameras simultaneously** using C++ multithreading.
* Supports both **CPU** and **CUDA (GPU)** inference.
* Detects objects and overlays bounding boxes with class labels.
* Tracks **active vehicles per frame** and maintains a global vehicle count.
* Uses **frame skipping** for improved performance on lower-end hardware.

---

## 🛠 Requirements

* **C++17 or later**
* **OpenCV 4.5+** (with DNN module enabled)
* **CUDA/cuDNN** (optional, for GPU acceleration)
* YOLOv5n ONNX model (`yolov5n.onnx`)
* Classes file (`classes.txt` with COCO class labels)

Install OpenCV (Linux example):

```bash
sudo apt-get install libopencv-dev
```

For CUDA build, ensure OpenCV is compiled with `-D WITH_CUDA=ON`.

---

## 🚀 Build Instructions

1. Clone this repository:

```bash
git clone https://github.com/NoobSaifii/vehicle-detection-yolov5.git
cd vehicle-detection-yolov5
```

2. Create a `build/` directory and compile:

```bash
mkdir build && cd build
cmake ..
make
```

3. Run the program:

```bash
./vehicle_detection         # Run on CPU
./vehicle_detection cuda    # Run on GPU (CUDA)
```

---

## 📂 Project Structure

```
vehicle-detection-yolov5/
│-- main.cpp                # Main source code
│-- classes.txt             # COCO class labels
│-- yolov5n.onnx            # YOLOv5 nano model (ONNX)
│-- CMakeLists.txt          # Build configuration
│-- README.md               # Documentation
```

---

## 📊 Example Output

* Bounding boxes drawn around detected vehicles.
* Per-frame vehicle count overlayed.
* Final output:

```
Total vehicles detected across all cameras: 235
```

---

## 📖 Dataset & Model

* Pre-trained **YOLOv5n (nano)** model exported to ONNX.
* COCO dataset labels used in `classes.txt`.

Download YOLOv5 ONNX model:

* [YOLOv5 Releases (Ultralytics)](https://github.com/ultralytics/yolov5/releases)

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use, modify, and share.
