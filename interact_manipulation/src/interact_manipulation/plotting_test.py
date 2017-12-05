import time
import matplotlib.pyplot as plt
import numpy as np
import math

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
    return get_memory(t), get_cpu(t)

def gaussian(mu, var):
    """ Compute a multivariate gaussian
        ----
        Param: mu a vector of means
        Param: var a scaler for equal variance on all dimensions
        ----
        Return: A function which computes the probability density of a 3D point
    """
    cov = var * np.eye(mu.shape[0])
    return lambda x: (1./np.sqrt(2*math.pi*np.linalg.det(cov))) * np.exp(
            -(1./2.) * np.dot(np.dot((x - mu), np.linalg.inv(cov)), (x - mu))
            )

def direction(x, y):
    """ Compute the direction from x to y
        Param: x a numpy array
        Param: y a numpy array
        Return: numpy array of unit length to y from x
    """
    return (y - x)/np.linalg.norm(y - x + 1e-12)

def normalize(beliefs):
    """ Normalize a descrete set of beliefs
        Param: beliefs, a list of scaler beliefs #TODO check this
    """
    return np.asarray(beliefs)/np.sum(np.asarray(beliefs))

def update(prev_eef_pos, eef_pos, goals, prior_beliefs):
    """ Updates the belief over goals
        Param: prev_eef_pos a (3,) numpy array with xyz of robot end effector
        Param: eef_pos current location of end effector
        Param: goals a list of (3,) numpy arrays with goals
        Param: prior_beliefs a list of prior beliefs over goals
        Return: a list of new beliefs given the observation (eef_pos)
    """
    goal_dirs = [direction(eef_pos,goal) for goal in goals]
    
    interaction_dir = direction(prev_eef_pos,eef_pos)

    beliefs = np.array([b*gaussian(goal_dir, 1e-2)(interaction_dir) for (b, goal_dir) in zip(prior_beliefs, goal_dirs)])
    return beliefs

def plot():
    fig, (ax1,ax2) = plt.subplots(1,2)
    ind = np.arange(1, 3)

    # show the figure, but do not block
    plt.show(block=False)

    #print(get_stats(0))
    g1, g2 = plt.bar(ind, get_stats(0))
    g1.set_facecolor('r')
    g2.set_facecolor('g')
    #pn.set_facecolor('b')
    ax1.set_xticks(ind)
    ax1.set_xticklabels(['Goal 1', 'Goal 2'])
    ax1.set_ylim([0, 100])
    ax1.set_ylabel('Belief')
    ax1.set_title('Robot Beliefs over goals')
    
    ax2.set_title('plot2')
    
    eef_positions = []
    for i in range(10):
        x = i/10
        y = i/10
        z = 0
        eef_positions.append(np.array([x,y,z]))

    for i in range(2):  # run for a little while
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
        fig.canvas.draw_idle()
        junk = raw_input("Belief update: {} new_eef_pos {}".format(i, eef_positions[i]))
        try:
            # make sure that the GUI framework has a chance to run its event loop
            # and clear any GUI events.  This needs to be in a try/except block
            # because the default implementation of this method is to raise
            # NotImplementedError
            fig.canvas.flush_events()
        except NotImplementedError:
            pass


def test_update():
    prev_eef_pos = np.array([1,1.5,0])
    eef_pos = np.array([1,2,0])
    goal1 = np.array([1,1,0])
    goal2 = np.array([0,0,0])
    goals = [goal1, goal2]
    beliefs = [0.5,0.5] #start off with 50/50 belief
    print(update(prev_eef_pos, eef_pos, goals, beliefs))


if __name__ == '__main__':
    plot()
    # if len(sys.argv) < 2:
    #     print "ERROR: need an experiment to run"
    # else:
    #     run(*parse(load(sys.argv[1])))
