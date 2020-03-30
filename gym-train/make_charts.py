import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import os


RUN_NUM = 8
EPISODES = 500
style.use('ggplot')

fig = plt.figure(figsize=(15, 8))
aggr = np.load(f"qtables_{RUN_NUM}/aggregated.npy", allow_pickle=True).item()
rewads = np.load(f"qtables_{RUN_NUM}/rewards.npy")

os.makedirs(f"qtables_{RUN_NUM}_charts", exist_ok=True)


def get_q_color(value, vals):
    if value == max(vals):
        return 'green', 1.0
    else:
        return 'red', 0.4


def save_chart(i, n):
    q_table = np.load(f"qtables_{RUN_NUM}/{n}-qtable.npy")
    for _x in q_table:
        for _y in _x:
            if np.argmax(_y) == 1:
                print(_y)
            _y[:] = _y == _y.max()
    interp = 'kaiser'

    ax1 = fig.add_subplot(221)
    plt.imshow(np.flip(q_table[:, :, 0], axis=1), cmap='GnBu', interpolation=interp)
    # fig1.patch.set_visible(True)
    ax2 = fig.add_subplot(222)
    plt.imshow(np.flip(q_table[:, :, 1], axis=1), cmap='GnBu', interpolation=interp)

    ax3 = fig.add_subplot(223)
    plt.imshow(np.flip(q_table[:, :, 2], axis=1), cmap='GnBu', interpolation=interp)

    ax4 = fig.add_subplot(224)

    x = aggr['ep'][0:i]
    y_max = aggr['max'][0:i]
    y_avg = aggr['avg'][0:i]
    y_min = aggr['min'][0:i]

    plt.plot(x, y_max, label='Max score')
    plt.plot(x, y_avg, label='Avg score')
    plt.plot(x, y_min, label='Min score')
    plt.scatter(range(n), rewads[:n], marker='o', c='k', s=10)
    plt.grid()

    ax1.title.set_text("Action 0")
    ax2.title.set_text("Action 1")
    ax3.title.set_text("Action 2")
    ax4.title.set_text("Best agent")

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.legend(loc=2)

    plt.savefig(f"qtables_{RUN_NUM}_charts/{n}.png")
    plt.clf()


for i, n in enumerate(range(0, EPISODES, 10)):
    i += 1
    print(f"Started: {i}, {n}")
    save_chart(i=i, n=n)



