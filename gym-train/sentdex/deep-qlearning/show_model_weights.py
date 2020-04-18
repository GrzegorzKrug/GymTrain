from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.compat.v1.Session(config=config)


MODEL_NAME = "Lin32-Drop0_2-Relu32-LinearOut-LR0_05-"
FIG_SIZE = (16, 9)
INPUT = 2
OUTPUT = 3


def show_model(input_weights: '2d list of layers weights'):
    fig = plt.figure(figsize=FIG_SIZE)
    # ax = fig.add_subplot(111)
    bias = False
    Bias = []
    Weights = []
    INPUT_NODES = [[0, 2], [0, -2]]
    Node_map = [INPUT_NODES]

    for index, layer in enumerate(input_weights):
        node_count = len(layer)
        if bias:
            Bias.append(layer)
            if not index == 0:
                Node_map.append([])
                for node_index, node in enumerate(layer):
                    x = index * 20
                    y_ratio = 2
                    y = node_index*y_ratio - node_count*y_ratio/2 + y_ratio/2
                    Node_map[-1].append([x, y])
        else:
            Weights.append(layer)
        bias ^= True

    for index, layer in enumerate(Weights):
        layer = Weights[index]
        for n_i, node in enumerate(layer):
            max_weight = np.max(node)
            min_weight = np.min(node)

            if index == 0:
                color = 'g'
            else:
                color = 'w'
            range_val = [-1, 1]
            for w_i, weight in enumerate(node):
                interp = np.interp(weight, [min_weight, max_weight], range_val)
                # print(interp)
                line_color = 'r' if interp < 0 else 'g'
                x = [Node_map[index][n_i][0], Node_map[index+1][w_i][0]]
                y = [Node_map[index][n_i][1], Node_map[index+1][w_i][1]]
                plt.plot(x, y, linewidth=abs(interp)/2, alpha=abs(interp)*0.8, color=line_color)

            plt.scatter(Node_map[index][n_i][0], Node_map[index][n_i][1], marker='o', s=150, edgecolors='k', c=color)

    for node in Node_map[-1]:
        plt.scatter(node[0], node[1], marker='o', s=150, edgecolors='k', c='r')

    dist = 64
    plt.xlim([-5, 2*dist])
    plt.ylim([-dist, dist])
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    plt.axis('off')

    os.makedirs(f"{MODEL_NAME}-weights", exist_ok=True)
    counter = 0
    pic_name = f"{MODEL_NAME}-weights/0.png"

    while os.path.isfile(pic_name):
        counter += 1
        pic_name = f"{MODEL_NAME}-weights/{counter}.png"
    plt.savefig(pic_name)
    # plt.show()
    return pic_name


def create_model():
    model = Sequential([
            # Flatten(),
            Dense(32, activation='linear', input_shape=(2,)),
            # Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(3, activation='linear')
    ])
    model.compile(optimizer=Adam(lr=0.001),
                  loss="mse",
                  metrics=['accuracy'])

    return model


model = create_model()
model.load_weights(f"models/{MODEL_NAME}")

weights = model.get_weights()
pic_name = show_model(weights)

print(f"Saved to {pic_name}")
