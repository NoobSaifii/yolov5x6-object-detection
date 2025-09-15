# YOLOv5n Real-Time Object Detection 🚀

A **C++ project** using **OpenCV DNN** and **YOLOv5n (ONNX)** to detect and count objects in real-time from multiple camera feeds.

---

## 📌 Features

* Loads **YOLOv5n** model for higher-accuracy detection.
* Supports **multi-camera feeds** with **threading**.
* Runs on **CPU** or **CUDA (GPU)** for faster inference.
* Detects objects and draws **bounding boxes** with class labels.
* Tracks **active objects per frame** and updates a **global object count**.
* Uses **frame skipping** to improve performance.
* Implements **letterboxing** in preprocessing for correct aspect ratio.

---

## 🛠 Requirements

* **C++17 or later**
* **OpenCV 4.5+** with DNN module
* **CUDA/cuDNN** (optional, for GPU acceleration)
* YOLOv5x6 ONNX model (`yolov5n.onnx`)
* Classes file (`classes.txt`)

Install OpenCV (Linux example):

```bash
sudo apt-get install libopencv-dev
```

For CUDA, ensure OpenCV is compiled with `-D WITH_CUDA=ON`.

---

## 🚀 Build Instructions

1. Clone the repository:

```bash
git clone https://github.com/NoobSaifii/yolov5n-object-detection.git
cd yolov5x6-object-detection
```

2. Create a `build/` directory and compile:

```bash
mkdir build && cd build
cmake ..
make
```

3. Run the program:

```bash
./object_detection          # CPU mode
./object_detection cuda     # GPU (CUDA) mode
```

---

## 📂 Project Structure

```
yolov5x6-object-detection/
│-- main.cpp                # Main detection code
│-- classes.txt             # Class labels file
│-- yolov5n.onnx           # YOLOv5x6 ONNX model
│-- CMakeLists.txt          # Build configuration
│-- README.md               # Project documentation
```

---

## 📊 Example Output

* Bounding boxes with class labels for each detected object.
* Active objects count displayed on frame.
* Final output:

```
Total Objects detected across all cameras: 342
```

---

## 📖 Model & Dataset

* Pre-trained **YOLOv5x6** model exported to ONNX.
* COCO dataset classes used in `classes.txt`.

Download YOLOv5n ONNX model:

* [YOLOv5 Releases - Ultralytics](https://github.com/ultralytics/yolov5/releases)

---

## 📜 License

This project is licensed under the **MIT License** – free to use, modify, and share.
