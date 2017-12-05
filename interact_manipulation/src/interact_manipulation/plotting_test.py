import time
import matplotlib.pyplot as plt
import numpy as np


def get_memory(t):
    "Simulate a function that returns system memory"
    return 100 * (0.5 + 0.5 * np.sin(0.5 * np.pi * t))


def get_cpu(t):
    "Simulate a function that returns cpu usage"
    return 100 * (0.5 + 0.5 * np.sin(0.2 * np.pi * (t - 0.25)))


def get_net(t):
    "Simulate a function that returns network bandwidth"
    return 100 * (0.5 + 0.5 * np.sin(0.7 * np.pi * (t - 0.1)))


def get_stats(t):
    return get_memory(t), get_cpu(t), get_net(t)

fig, ax = plt.subplots()
ind = np.arange(1, 2)

# show the figure, but do not block
plt.show(block=False)


g1, g2 = plt.bar(ind, get_stats(0))
g1.set_facecolor('r')
g2.set_facecolor('g')
#pn.set_facecolor('b')
#ax.set_xticks(ind)
ax.set_xticklabels(['Goal 1', 'Goal 2'])
ax.set_ylim([0, 100])
ax.set_ylabel('Belief')
#ax.set_title('System Monitor')

for i in range(200):  # run for a little while
    m, c, n = get_stats(i / 10.0)

    # update the animated artists
    g1.set_height(m)
    g2.set_height(c)

    # ask the canvas to re-draw itself the next time it
    # has a chance.
    # For most of the GUI backends this adds an event to the queue
    # of the GUI frameworks event loop.
    fig.canvas.draw_idle()
    try:
        # make sure that the GUI framework has a chance to run its event loop
        # and clear any GUI events.  This needs to be in a try/except block
        # because the default implementation of this method is to raise
        # NotImplementedError
        fig.canvas.flush_events()
    except NotImplementedError:
        pass