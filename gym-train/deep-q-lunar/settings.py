SIM_COUNT = 10
EPOCHS = 2000
TRAIN_MAX_MIN_DURATION = 40

SHOW_FIRST = True
SHOW_EVERY = 25
RENDER_DELAY = 0.0003

REPLAY_MEMORY_SIZE = 5 * 200 * SIM_COUNT  # 10 full games 3k each
MIN_BATCH_SIZE = 100
MAX_BATCH_SIZE = 5000

# Training method
ALLOW_TRAIN = True
LOAD_MODEL = True

model = "StepModel"
MODEL_NAME = f"{model}-18-100"

DENSE1 = 1024
DENSE2 = 1024
# Training params
ALPHA = 1e-4
BETA = 1e-6

DISCOUNT = 0.95
FIRST_EPS = 0.3
RAMP_EPS = 0.4
INITIAL_SMALL_EPS = 0.2
END_EPS = 0
EPS_INTERVAL = 20

# Settings
SAVE_PICS = ALLOW_TRAIN
SHOW_LAST = False
# PLOT_ALL_QS = True
# PLOT_FIRST_QS = False
# COMBINE_QS = True
SOUND_ALERT = True
