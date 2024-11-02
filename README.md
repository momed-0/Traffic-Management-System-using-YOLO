# 🚦 Traffic Management System Using IoT and Machine Learning


**The Traffic Management System Using IoT and Machine Learning** project aims to enhance urban traffic management by detecting real-time vehicle flow and analyzing the data to reduce congestion. Utilizing a Jetson Nano with a YOLO object detection model, the system detects vehicles such as cars, bikes, and autorickshaws. The vehicle data is transmitted to the cloud using the MQTT protocol, interfacing with AWS IoT Core. The long-term goal is to leverage this data to build a predictive model for traffic patterns and integrate it into a digital twin using VSim software.

---

## 📑 Project Overview

This project focuses on improving traffic flow by integrating IoT devices and machine learning algorithms. A Jetson Nano device running the YOLO object detection model identifies different types of vehicles in real-time at a 4-way intersection. Data is sent to the cloud using AWS IoT Core, where it is stored and analyzed for traffic optimization. Future work includes determining the format of data for VSim import and defining data handling workflows in AWS.

---

## 🛠️ Technologies Used

- **Hardware**: Jetson Nano, OAK-D Lite Camera
- **Machine Learning**: YOLO Object Detection, DeepSort
- **Cloud**: AWS IoT Core ( AWS Free Tier)
- **Software**: VSim, Python, Ubuntu

---

## 🚧 Project Roadmap

### 🖥️ Jetson Nano and YOLO Implementation
- [x]  **Acquire Jetson Nano hardware**
  _Status: Completed_ ✅
- [x] **Train YOLO model on custom dataset**  
  _Status: Completed_ ✅
- [x] **Integrate DeepSort into YOLO**  
  _Status: Completed_ ✅
- [ ] **Extend the model to run detections on a 4-way junction (using a stock video)**
  _Status: Open_ 🟢
- [ ] **Check compatibility of module dependencies with Jetson Nano**
  _Status: Open_ 🟢
- [ ] **Check if its possible to Upgrade Jetson Nano to at least Ubuntu 20.04**  
  _Status: Open_ 🟢
- [ ] **Run YOLO on Jetson Nano for real-time detection (using a stock video)**
  _Status: Open_ 🟢
- [ ] **Integrate OAK-D Lite Camera on Jetson Nano**
  _Status: Open_ 🟢
- [ ] **Perform real-time vehicle detection using the live camera feed**
  _Status: Open_ 🟢

### ☁️ Cloud Integration
- [x] **Sign up for AWS free tier and obtain credits**  
  _Status: Completed_ ✅
- [x] **Design cloud architecture in AWS**  
  _Status: Completed_ ✅
- [x] **Register Jetson Nano in AWS IoT Core and establish a connection**  
  _Status: Completed_ ✅
- [ ] **Send detection results from Jetson Nano to AWS IoT Core**
  _Status: Open_ 🟢
- [ ] **Define data format and transmission frequency to AWS IoT Core**
  _Status: Open_ 🟢
- [ ] **Decide post-receipt processing of data in AWS IoT Core**
  _Status: Blocked_ 🔴



## 📁 Repository Structure
```plaintext
- /deep_sort       # Original deepsort code forked from repository and did modifications according to out dependencies
- /main.py     # python script for object detection
