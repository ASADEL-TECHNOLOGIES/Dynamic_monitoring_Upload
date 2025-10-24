# 🧠 Dynamic Monitoring

## 📌 Overview

**Dynamic Monitoring** is an intelligent, multi-camera real-time object detection and tracking system.  
It supports **dynamic Region of Interest (ROI)** setup, **multi-threading** and **multi-processing**, and runs on either **GPU** or **CPU** — fully configurable via a single configuration file.

The system is designed for scalable surveillance or industrial monitoring setups that require simultaneous camera feeds (video files, RTSP streams, or webcams).

---

## 🎯 Main Agenda

The main goal of **Dynamic Monitoring** is to create a **fully dynamic and configurable object detection pipeline** built on **YOLO**.  
This project allows users to:

- 🧩 **Switch YOLO models dynamically** — simply update the model path in the config (no code change needed).  
- 🎯 **Modify or add detection classes** directly in `config.json`.  
- 📍 **Create custom ROIs per camera** interactively to focus detection areas.  
- 📊 **Get real-time detections and object counts** per interval from defined ROIs.  
- 🔄 **Run detection on multiple cameras simultaneously** using multiprocessing or threading.  

In short — you can **plug in any YOLO model, any class, and any number of cameras**, define your ROIs, and get instant detection counts.

---

## ⚙️ Key Features

- 📸 **Multi-camera support** — works seamlessly with multiple sources (videos, RTSP, webcam).  
- 🚀 **Supports both multiprocessing & multithreading** — configurable directly from `config.json`.  
- 🧩 **Dynamic ROI management** — interactive region selection for per-class monitoring.  
- 💪 **GPU/CPU flexibility** — switch inference device anytime without changing code.  
- 📊 **MySQL database integration** — automatic count logging at fixed time intervals.  
- ⚡ **Frame skip control** — allows frame-level optimization for faster inference.  
- 🧠 **Supports YOLO + ByteTrack tracking pipeline** for real-time performance.  

---

## 🧰 Setup & Usage

### 1️⃣ Create the Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2️⃣ Setup ROIs for Each Camera

```python3 main/setup_rois.py```

If a ROI file already exists for a camera, you will be prompted to redraw it (y/n).
Select the ROI interactively by drawing rectangles on the frame.
ROIs are saved automatically in /rois/<model_name>/<camera_name>.json.

3️⃣ Run All Cameras

```python3 main/start_all.py```

Reads all camera configurations from config.json.
Switch between multiprocessing or threading via the mode field in the config.
Toggle GPU or CPU usage by setting gpu = true/false in the config.
Counts are logged automatically into the database at configured intervals.

4️⃣ Add or Modify Cameras
To add a new camera or modify an existing one, update config.json with the new source, name, or type (video, RTSP, webcam).
ROIs for new cameras can be created using setup_rois.py.

Folder Overview
main/ — core scripts to run cameras, YOLO detection, and ROI setup.
rois/ — stores ROI JSON files per camera and model.
config/ — contains config.json for camera sources, model path, classes, frame skip, hardware selection, and multiprocessing/threading mode.
db/ — database utilities for connection and count logging.
roi_handler/ — utilities to interactively select, save, and load ROIs.
videos/ — stores video feeds for testing.
models/ — YOLO detection models.
venv/ — Python virtual environment with dependencies.

## 📊 Observations

### Multiprocessing (GPU)
- GPU utility: **65% average**
- Memory usage for 6 cameras (4 videos, 1 RTSP, 1 webcam): **1,846 MiB (~1.8 GB)**
- CPU load: **average 5.44**
- Memory usage: **8.27 GB**
- Camera counts per 30-sec interval:
  - 6 cameras (4 videos, 1 RTSP, 1 webcam) → **11 counts**
  - 5 cameras (3 videos, 1 RTSP, 1 webcam) → **11 counts**
  - 4 cameras (2 videos, 1 RTSP, 1 webcam) → **13 counts**
  - 3 cameras (2 videos, 1 RTSP) → **13 counts**
  - 3 cameras (2 videos, 1 webcam) → **16 counts**
- All 6 cameras run fine on GPU simultaneously.

### Threading
- GPU utility: **656 MiB**
- CPU load: **average 4**
- Memory usage: **5.6 GB**

### Config Flexibility
- Switch between **multiprocessing** or **threading** anytime from `config.json`.
- Switch between **GPU** or **CPU** anytime from `config.json`.
