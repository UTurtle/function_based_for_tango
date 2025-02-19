import os

# 만약 환경변수에 값이 있다면 그 값을 사용하고, 없으면 기본값을 사용한다.
OUTPUT_DIR = os.environ.get("TANGO_OUTPUT_DIR", "augmentation/output")
DATASET_OUTPUT_FOLDER = os.environ.get("TANGO_DATASET_OUTPUT_FOLDER", "data")
DATASET_NAME = 'function_based'
DATASET_PATH = f'{DATASET_OUTPUT_FOLDER}/{DATASET_NAME}'

# 나머지 상수들은 그대로...

MAX_LEVEL = 2
TOTAL_FILES = 10  # dataset size
SR = 16000
DURATION = 10.0
N_FFT = 256
HOP_LENGTH = 256
WINDOW = "hann"

NOISE_PIPE_LINE_DEBUG_LOGGING = True
LOGGER_DIR = 'logs'
