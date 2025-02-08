# 🚦 Traffic Management System Using IoT and Machine Learning

## Pre-Requisites needed to build the system
Assuming you are using Nvidia Jetson Orin Nano.

This setup needs atleast JetPack 6.0 (Preferred JetPack 6.1).
You need atleast a 64GB SD card (128GB preferred)
Install JetPack using sdkmanager by following this [guide](https://www.jetson-ai-lab.com/initial_setup_jon.html)

If you don't have an display , after pre-config user settings from sdkmanager.Connect a USB Type-C cable from target to host and then ssh like below.

```bash
ssh user_name@192.168.55.1
```

You have to build OpenCV with CUDA support. (Try [this](https://qengineering.eu/install-opencv-on-jetson-nano.html)
Ensure that you have gstreamer support . (If your OpenCV isn't compiled with gstreamer , use normal video capture code to process the frame)
Install pytorch and torchvision from [here](https://developer.nvidia.com/embedded/downloads) . 
Note: You might have to downgrade numpy to a version less than 2 and you have to find torchvision compatible with current pytorch (and jetpack).

How to convert the given best.pt model to best.engine model (specific to each hardware)? 
Try this [link] (https://github.com/triple-Mu/YOLOv8-TensorRT)

How to redirect the dispaly to host when we are using ssh ?

Find $DISPLAY on host and export it on ssh

```bash
(host) echo $DISPLAY
(ssh) jetson@ubuntu:~$ export DISPLAY=:1
```

Register a thing on AWS IoT Core and under connect a device wizard follow the steps and install aws-iot-sdk-v2
Also download the certificates and wallet from IoT core and extract it to keys/ folder in the repository.


## Development Flow

 - main: default branch 

 - v1-jetson-orin(dev): Push to this branch and from this branch create a PR to merge it to main. code for jetson orin (JetPack 6.0 and libraries optimized for latest python version (3.10) )

 - v1-jetson-nano: code for jetson nano (JetPack < 5 and libraries optimized for python 3.6) .Do not support deep_sort tracking
