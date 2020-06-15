SIM_COUNT = 5
EPOCHS = 500
TRAIN_MAX_MIN_DURATION = 10
TIMEOUT_AGENT = 90  # seconds

SHOW_FIRST = True
SHOW_INTERVAL = 1  # in minutes
RENDER_DELAY = 0.0001
RENDER_WITH_ZERO_EPS = True

# Step training
REPLAY_MEMORY_SIZE = 10_000 * SIM_COUNT
MIN_BATCH_SIZE = 10 * SIM_COUNT
MAX_BATCH_SIZE = 3 * MIN_BATCH_SIZE
CLEAR_MEMORY_AFTER_TRAIN = True

# Training method
ALLOW_TRAIN = True
STEP_TRAIN = True
LOAD_MODEL = True
SAVE_MODEL = True

MODEL_NAME = f"Model-9"
DENSE1 = 1000
DENSE2 = 1000
DROPOUT1 = 0.1
DROPOUT2 = 0.2

# Training params
ALPHA = 1e-3
BETA = 1e-2
GAMMA = 0.98

ENABLE_EPS = False
FIRST_EPS = 0.3
RAMP_EPS = 0.4
INITIAL_SMALL_EPS = 0.03
END_EPS = 0
EPS_INTERVAL = 20

# Settings
SAVE_PICS = ALLOW_TRAIN
SHOW_LAST = False
SOUND_ALERT = True
# PLOT_ALL_QS = True
# PLOT_FIRST_QS = False
# COMBINE_QS = True


ACTION_SPACE = 2  # Turn left, right or none
INPUT_SHAPE = (8,)
