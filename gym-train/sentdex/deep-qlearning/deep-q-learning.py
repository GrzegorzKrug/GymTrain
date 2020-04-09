from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2d, Activation, Flatten
from keras.callbacks import TensorBoard


class DQNAgent:
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPolling2D(2, 2))
