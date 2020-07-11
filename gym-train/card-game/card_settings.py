MODEL_NAME = "30c"
GAME_NUMBER = 1000
TRAIN_TIMEOUT = 30 * 60
LOSSFN = 'mse'  # Huber loss?
SIM_COUNT = 10

"Train Params"
ALPHA = 1e-4  # 10e-4 def #  60e-4 too fast?
EPS = 0.5
EPS_INTERVAL = 200
EPS_BIAS = 25
EPS_DIVIDE = 5

"Model"
LAYERS = (2000, 0.001, 200)
INPUT_SHAPE = (98 * 8 + 4,)

"Settings"
ALLOW_TRAIN = True
STEP_TRAIN = False
PLOT_AFTER = True
TRAIN_EVERY = 1
MIN_BATCH_SIZE = 50
MAX_BATCH_SIZE = MIN_BATCH_SIZE
BATCH_SIZE = 50
MEMOR_MAX_SIZE = 100 * SIM_COUNT * 100
CLEAR_MEMORY = False
DISCOUNT = 0.9

"Env"
ACTION_SPACE = 32
GAME_TIMEOUT = 150
"Rewards"
SKIP_MOVE = -0.1
GOOD_MOVE = -1
LOST_GAME = -100
INVALID_MOVE = -150

"Plot"
GRAPH_CUT_AT = 100  # New plot when x is reached

DEBUG = False
SAVE_INTERVAL = 5 * 60
