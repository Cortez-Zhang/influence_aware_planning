import matplotlib.pyplot as plt
import numpy as np

def get_memory(t):
    "Simulate a function that returns system memory"
    return 100 * (0.5 + 0.5 * np.sin(0.5 * np.pi * t))


def get_cpu(t):
    "Simulate a function that returns cpu usage"
    return 100 * (0.5 + 0.5 * np.sin(0.2 * np.pi * (t - 0.25)))


def get_stats(t):
    return get_memory(t), get_cpu(t)

plt.ion()

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

plot = ax.scatter([], [])
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

ind = np.arange(1, 3)
g1, g2 = plt.bar(ind, get_stats(0))
ax2.set_xticks(ind)
ax2.set_xticklabels(['Goal 1', 'Goal 2'])
ax2.set_ylim([0, 100])
ax2.set_ylabel('Belief')
ax2.set_title('Robot Beliefs over goals')

#g1, g2 = plt.bar(ind, get_stats(0))

for i in range(20):
    # get two gaussian random numbers, mean=0, std=1, 2 numbers
    point = np.random.normal(0, 1, 2)
    # get the current points as numpy array with shape  (N, 2)
    array = plot.get_offsets()

    # add the points to the plot
    array = np.append(array, point)
    plot.set_offsets(array)

    # update x and ylim to show all points:
    ax.set_xlim(array[:, 0].min() - 0.5, array[:,0].max() + 0.5)
    ax.set_ylim(array[:, 1].min() - 0.5, array[:, 1].max() + 0.5)
    # update the figure
    fig.canvas.draw()

    ####belief update code#############


    m, c = get_stats(i / 10.0)  #TODO test pause capability
                        #TODO plot the positions one at a time along with the belief updates 

    # update the animated artists
    g1.set_height(m)
    g2.set_height(c)

    #fig1 = plt.figure(1)
    #plt.scatter(eef_positions[i][0],eef_positions[i][1], 'ro')

    # ask the canvas to re-draw itself the next time it
    # has a chance.
    # For most of the GUI backends this adds an event to the queue
    # of the GUI frameworks event loop.
    fig2.canvas.draw_idle()
    junk = raw_input("Belief update: {} new_eef_pos".format(i))
    try:
        # make sure that the GUI framework has a chance to run its event loop
        # and clear any GUI events.  This needs to be in a try/except block
        # because the default implementation of this method is to raise
        # NotImplementedError
        fig2.canvas.flush_events()
    except NotImplementedError:
        pass
