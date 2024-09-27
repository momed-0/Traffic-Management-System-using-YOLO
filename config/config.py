import os

# Paths to necessary files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')
ENCODER_MODEL_PATH = os.path.join(BASE_DIR, 'deep_sort', 'networks', 'mars-small128.pb')

VIDEO_PATH = os.path.join(BASE_DIR, 'data', 'id4.mp4')
CLASS_FILE = os.path.join(BASE_DIR, 'data', 'coco1.txt')

# DeepSort parameters
MAX_COSINE_DISTANCE = 0.4
NN_BUDGET = None

# Tracking line parameters
CY1 = 427
OFFSET = 6
