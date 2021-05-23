# data config
TRAIN_PATH = "./new_train/new_train"
TEST_PATH = "./new_val_in/new_val_in"
SUBMISSION_FILE = "./sample_submission.csv"
SUBMISSION_PATH = "./submissions"

# training config
TRAIN_SIZE = 0.9
VAL_SIZE = 0.1

LEARNING_RATE = 1e-3

NUM_EPOCH = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EARLY_STOP_MAX = 6

# model config 
INPUT_SIZE = 5
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 512
OUTPUT_SIZE = 2

# feature engineering configs
NEARBY_DISTANCE_THRESHOLD = 50.0  # Distance threshold to call a track as neighbor
DEFAULT_MIN_DIST_FRONT_AND_BACK = 100. # default distance 