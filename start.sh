# Ensure the script is run with sudo

sudo jetson_clocks

export DISPLAY=:0

python3 main.py --model_path=./models/yolov11/best_yolov11.engine --class_list=./config/yolov11/class.txt --zone_name=test --video_path=./video/test_footage.mp4 --publish_interval=2 --line_y=600 --publish
