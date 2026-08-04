# AI Computer Vision Benchmark Platform - Intel NUC

Edge AI benchmarking and monitoring platform running on Intel NUC.

The system provides a complete environment for testing Computer Vision models,
collecting performance metrics and visualizing results through a web dashboard.

Intel NUC acts as the main edge server responsible for benchmark orchestration,
data collection and visualization.

---

## Overview

This project was created to evaluate Computer Vision workloads on edge devices.

Main goals:

- benchmark YOLO object detection models
- measure real-time inference performance
- collect hardware utilization metrics
- visualize benchmark results
- support distributed edge AI architecture

---

## System Architecture


Camera / Video Source
          |
          |
          v

+----------------------+
|     Intel Joule      |
|                      |
| YOLO Inference Node  |
| AI Worker            |
+----------------------+

          |
          |
          v

+----------------------+
|      Intel NUC       |
|                      |
| Benchmark Server     |
| Dashboard            |
| Metrics Storage      |
+----------------------+

          |
          |
          v

       Web Interface


---

## Features

### Computer Vision Benchmarking

Supported models:

- YOLOv8
- YOLO11
- YOLO26


Capabilities:

- object detection
- FPS measurement
- inference latency measurement
- video stream processing
- benchmark comparison


---

## Monitoring

Collected metrics:

- CPU utilization
- RAM usage
- inference time
- FPS
- processed frames
- detection statistics
- benchmark history


---

## Web Dashboard

The platform provides:

- benchmark visualization
- system monitoring
- experiment tracking
- performance comparison


---

## Technology Stack

Software:

- Python
- Flask
- OpenCV
- Ultralytics YOLO
- InfluxDB
- HTML/CSS/JavaScript


Hardware:

- Intel NUC
- Intel Joule
- IP cameras
- ESP32 cameras


---

## Project Structure


.
├── benchmark.py
├── benchmark_models.py
├── camera_reader.py
├── config.py
├── influx_writer.py
├── server.py
├── utils.py
│
├── web/
│   ├── app.py
│   ├── status.py
│   └── templates/
│
├── requirements.txt
└── README.md


---

## Installation


Clone repository:

git clone git@github.com:mateusz-formela/ai-computer-vision-benchmark-platform-nuc.git


Create virtual environment:

python3 -m venv venv

source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


---

## Running


Start benchmark:

python benchmark.py


Start web server:

python server.py


Dashboard:

http://NUC_IP:8080


---

## Applications

Possible applications:

- Edge AI research
- Computer Vision benchmarking
- autonomous systems
- robotics
- industrial monitoring
- smart camera systems


---

## Purpose

This project demonstrates:

- AI inference on edge devices
- distributed Computer Vision architecture
- embedded AI optimization
- performance benchmarking


---

## Author

Mateusz Formela

Electronics and Telecommunications Engineer

GitHub:
https://github.com/mateusz-formela
