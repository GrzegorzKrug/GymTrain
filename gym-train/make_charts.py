import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import threading

# style.use('ggplot')

RUN_NUM = 5
EPISODES = 20000


def get_q_color(value, vals):
    if value == max(vals):
        return 'green', 1.0
    else:
        return 'red', 0.4


fig = plt.figure(figsize=(15, 5))


def save_chart(n):
    q_table = np.load(f"qtables_{RUN_NUM}/{n}-qtable.npy")
    for _x in q_table:
        for _y in _x:
            _y[:] = _y == _y.max()
    interp = 'kaiser'
    # interp = 'mitchell'

    ax1 = fig.add_subplot(131)
    plt.imshow(q_table[:, :, 0], cmap='GnBu', interpolation=interp)
    ax2 = fig.add_subplot(132)
    plt.imshow(q_table[:, :, 1], cmap='GnBu', interpolation=interp)
    ax3 = fig.add_subplot(133)
    plt.imshow(q_table[:, :, 2], cmap='GnBu', interpolation=interp)

    ax1.set_xlabel("Action 0")
    ax2.set_xlabel("Action 1")
    ax3.set_xlabel("Action 2")

    plt.savefig(f"qtables_{RUN_NUM}_charts/{n}.png")
    plt.clf()


threads = []
for i in range(0, EPISODES, 10):
    print(f"Started: {i}")
    # th = threading.Thread(target=lambda: save_chart(n=i))
    # threads.append(th)
    # th.start()
    save_chart(i)

    # if len(threads) >= 5:
    #     for th in threads:
    #         th.join()
    #     threads = []



