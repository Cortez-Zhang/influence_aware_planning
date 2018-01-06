import time
import matplotlib.pyplot as plt
import numpy as np
import math
from jaco_moving_target import GoalInference
#from scipy.stats import multivariate_normal

goal1 = np.array([-0.4,0.1,0])
goal2 = np.array([-0.3,0,0])
goals = [goal1, goal2]
goal_inference = GoalInference(goals,variance = 1)

eef_positions = [np.array([-0.4,0,0]), np.array([-0.4, 0.025,0]),np.array([-0.4,0.05,0]), np.array([-0.4, 0.075 ,0]), np.array([-0.4, 0.05,0]), np.array([-0.36,0.05,0]), np.array([-0.35, 0.035,0]),np.array([-0.34, 0.025,0]), np.array([-0.33,0.02,0]),np.array([-0.32,0.01,0])]

def plot():
    plt.ion()
    fig_scatter, ax_scatter = plt.subplots()

    fig_bchart, ax_bchart = plt.subplots()


    plot = ax_scatter.scatter([], [])
    ax_scatter.set_xlim(-5, 5)
    ax_scatter.set_ylim(-5, 5)

    ind = np.arange(1, 3)
    g1, g2 = plt.bar(ind, [.5,.5])
    g1.set_facecolor('r')
    g2.set_facecolor('g')
    
    ax_bchart.set_xticks(ind)
    ax_bchart.set_xticklabels(['Goal 1', 'Goal 2'])
    ax_bchart.set_ylim([0, 100])
    ax_bchart.set_ylabel('Belief')
    ax_bchart.set_title('Robot Beliefs over goals')
    

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
        #beliefs = update(eef_positions[i-1],eef_positions[i],goals,beliefs)
        
        #update beliefs
        goal_inference.update(eef_positions[i],eef_positions[i-1])
        beliefs = goal_inference.current_beliefs

        # update the animated artists
        print("beliefs {}".format(beliefs))
        g1.set_height(beliefs[0]*100)
        g2.set_height(beliefs[1]*100)

        # ask the canvas to re-draw itself the next time it
        # has a chance.
        # For most of the GUI backends this adds an event to the queue
        # of the GUI frameworks event loop.
        fig_bchart.canvas.draw_idle()
        try:
            # make sure that the GUI framework has a chance to run its event loop
            # and clear any GUI events.  This needs to be in a try/except block
            # because the default implementation of this method is to raise
            # NotImplementedError
            fig_bchart.canvas.flush_events()
        except NotImplementedError:
            pass
        junk = raw_input("Belief update: {} new_eef_pos".format(i))

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
