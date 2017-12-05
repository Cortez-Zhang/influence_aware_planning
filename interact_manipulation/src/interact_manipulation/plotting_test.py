import time
import matplotlib.pyplot as plt
import numpy as np
import math
#from scipy.stats import multivariate_normal

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
        Params
        ----
        mu: a vector of means
        var: a scaler for equal variance on all dimensions
        Returns
        ----
        gaussian: A function which computes the probability density of a 3D point
    """
    cov = var * np.eye(mu.shape[0])
    return lambda x: (1./np.sqrt(2*math.pi*np.linalg.det(cov))) * np.exp(
            -(1./2.) * np.dot(np.dot((x - mu), np.linalg.inv(cov)), (x - mu))
            )

def direction(x, y):
    """ Compute the direction from x to y
        Params
        ---
        x: a numpy array
        y: a numpy array
        Returns
        ---
        normalized direction: numpy array of unit length to y from x
    """
    return (y - x)/np.linalg.norm(y - x + 1e-12)

def normalize(beliefs):
    """ Normalize a descrete set of beliefs
        Params
        ---
        beliefs: a list of scaler beliefs #TODO check this
        Returns
        ---
        norm_beliefs: a np array of normalized beliefs
    """
    return np.asarray(beliefs)/np.sum(np.asarray(beliefs))

def update(prev_eef_pos, eef_pos, goals, prior_beliefs):
    """ Updates the belief over goals
        Params
        ---
        prev_eef_pos: a (3,) numpy array with xyz of robot end effector
        eef_pos: current location of end effector
        goals: a list of (3,) numpy arrays with goals
        prior_beliefs: a list of prior beliefs over goals
        Returns
        ---
        norm_beliefs: a list of new beliefs given the observation (eef_pos)
    """
    print("prev_eef_pos {} curr_eef_pos {}".format(prev_eef_pos,eef_pos))
    goal_dirs = [direction(eef_pos,goal) for goal in goals]
    print("direction to goals: {}".format(goal_dirs))
    
    interaction_dir = direction(prev_eef_pos,eef_pos)
    print("interaction_dir: {}".format(interaction_dir))
    
    beliefs = np.array([b*gaussian(goal_dir, 1e-2)(interaction_dir) for (b, goal_dir) in zip(prior_beliefs, goal_dirs)])
    print("beliefs {}".format(beliefs))
    norm_belief = normalize(beliefs)
    print("normalized beliefs {}".format(norm_belief))
    return norm_belief
#def plot()

def plot():
    plt.ion()
    fig_scatter, ax_scatter = plt.subplots()

    fig_bchart, ax_bchart = plt.subplots()

    goal1 = np.array([1,1,0])
    goal2 = np.array([-1,-1,0])
    goals = [goal1, goal2]

    plot = ax_scatter.scatter([], [])
    ax_scatter.set_xlim(-5, 5)
    ax_scatter.set_ylim(-5, 5)

    ind = np.arange(1, 3)
    g1, g2 = plt.bar(ind, get_stats(0))
    g1.set_facecolor('r')
    g2.set_facecolor('g')
    
    ax_bchart.set_xticks(ind)
    ax_bchart.set_xticklabels(['Goal 1', 'Goal 2'])
    ax_bchart.set_ylim([0, 100])
    ax_bchart.set_ylabel('Belief')
    ax_bchart.set_title('Robot Beliefs over goals')
    
    eef_positions = []
    #Synthesize the data
    for i in range(20):
        x = i/10.0
        y = i/10.0
        z = 0
        eef_positions.append(np.array([x,y,z]))
    
    print(eef_positions)

    beliefs = [0.5,0.5]
    num_waypoints = 20
    for i in range(num_waypoints):
        # get the current points as numpy array with shape  (N, 2)
        array = plot.get_offsets()
        array = np.append(array, eef_positions[i][0:2])
        plot.set_offsets(array)

        # update x and ylim to show all points:
        ax_scatter.set_xlim(array[:, 0].min() - 0.5, array[:,0].max() + 0.5)
        ax_scatter.set_ylim(array[:, 1].min() - 0.5, array[:, 1].max() + 0.5)
        # update the figure
        fig_scatter.canvas.draw()

        ####belief update code#############
        beliefs = update(eef_positions[i-1],eef_positions[i],goals,beliefs)

        # update the animated artists
        print("beliefs {}".format(beliefs[0]))
        g1.set_height(beliefs[0]*100)
        g2.set_height(beliefs[1]*100)

        # ask the canvas to re-draw itself the next time it
        # has a chance.
        # For most of the GUI backends this adds an event to the queue
        # of the GUI frameworks event loop.
        fig_bchart.canvas.draw_idle()
        junk = raw_input("Belief update: {} new_eef_pos".format(i))
        try:
            # make sure that the GUI framework has a chance to run its event loop
            # and clear any GUI events.  This needs to be in a try/except block
            # because the default implementation of this method is to raise
            # NotImplementedError
            fig_bchart.canvas.flush_events()
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

def test_gaussian():

    goal_dir = np.array([1,0,0])
    interaction_dir = np.array([1,0,0])
    #print(multivariate_normal.pdf(interaction_dir,mean=goal_dir,covariance=interaction_dir))
    print(gaussian(goal_dir, 1e-2)(interaction_dir))
    
if __name__ == '__main__':
    #test_gaussian()
     plot()
    # if len(sys.argv) < 2:
    #     print "ERROR: need an experiment to run"
    # else:
    #     run(*parse(load(sys.argv[1])))
