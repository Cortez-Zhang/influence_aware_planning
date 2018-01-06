import time
import matplotlib.pyplot as plt
import numpy as np
import math

def plot():
    goal1 = np.array([-0.4,-0.1,0.538])
    goal2 = np.array([-0.2,-0.2,0.538])
    human_start = np.array([-.3,0.3,0.538])

    with open('go_fast_human_positions_local.np','r') as f:
        ex_human_positions = np.load(f)

    with open('go_fast_robot_positions_local.np','r') as f:
        ex_robot_positions = np.load(f)
        
    with open('go_fast_beliefs_local.np','r') as f:
        ex_robot_beliefs = np.load(f)
    
    with open('go_fast_velocities_local.np','r') as f:
        ex_human_velocities = np.linalg.norm(np.load(f),axis=1)

    
    with open('no_affect_human_positions.np','r') as f:
        no_human_positions = np.load(f)

    with open('no_affect_robot_positions.np','r') as f:
        no_robot_positions = np.load(f)
        
    with open('no_affect_beliefs.np','r') as f:
        no_robot_beliefs = np.load(f)
    
    with open('no_affect_velocities.np','r') as f:
        no_human_velocities = np.linalg.norm(np.load(f),axis=1)


    with open('go_slow_human_positions.np','r') as f:
        human_positions = np.load(f)

    with open('go_slow_robot_positions.np','r') as f:
        robot_positions = np.load(f)

    with open('go_slow_beliefs.np','r') as f:
        robot_beliefs = np.load(f)
    
    with open('go_slow_velocities.np','r') as f:
        human_velocities = np.linalg.norm(np.load(f),axis=1)

    fig_bchart, ax_pos = plt.subplots()
    ax_pos.plot(human_positions[:,0],human_positions[:,1],lw=3, c='#F79646', label="H Go Slow")
    ax_pos.plot(robot_positions[:,0],robot_positions[:,1], lw=3, c='#F79646',linestyle="--", label="R Go slow")
    
    ax_pos.plot(ex_human_positions[:,0],ex_human_positions[:,1],lw=3, c='#4BACC6', label="H Go Fast")
    ax_pos.plot(ex_robot_positions[:,0],ex_robot_positions[:,1],lw=3, c='#4BACC6',linestyle="--", label="R Go Fast")

    goal2h = ax_pos.plot(no_human_positions[:,0],no_human_positions[:,1],lw=3, c='#BFBFBF', label="H Baseline")
    goal1h = ax_pos.plot(no_robot_positions[:,0],no_robot_positions[:,1],lw=3, c='#BFBFBF',linestyle="--", label="R Baseline")
    
    font = {'weight': 'normal',
        'size': 16,
        }

    ax_pos.scatter(goal1[0],goal1[1],c='r',marker='x',s=50)
    ax_pos.scatter(goal2[0],goal2[1],c='g',marker='x',s=50)
    ax_pos.text(goal1[0]+.01, goal1[1], 'Goal 1', fontdict=font)
    ax_pos.text(goal2[0]+.01, goal2[1], 'Goal 2', fontdict=font)
    
    #print("goal1h {}".format(goal1h))
    first_legend = ax_pos.legend(bbox_to_anchor=(0.,.97,1.,.102),loc=3,ncol=3,mode="expand",borderaxespad=0.)

    ax_pos.set_ylabel('Y position (m)')
    ax_pos.set_xlabel('X position (m)')
    #ax_pos.set_title('Human & Robot Trajectories')

    time = np.linspace(0, 20*.02, num=len(robot_beliefs[:,0]), endpoint=True)

    fig_bel, ax_bel = plt.subplots()
    print(np.shape(robot_beliefs))
    print(np.shape(ex_robot_beliefs))
    print(np.shape(no_robot_beliefs))
    ax_bel.plot(time,robot_beliefs[:,0]*100, label='Go slow',c='#F79646', lw=3)
    ax_bel.plot(time[0:-1],ex_robot_beliefs[:,0]*100, label='Go Fast',c='#4BACC6',lw=3)
    ax_bel.plot(time,no_robot_beliefs[:,0]*100, label='Baseline',c='#BFBFBF',lw=3)

    #ax_bel.set_title('Human Beliefs of Robots goal')
    ax_bel.set_xlabel('Time (Seconds)')
    ax_bel.set_ylabel('Belief (Percent)')
    ax_bel.legend(loc=1)

    fig_vel, ax_vel = plt.subplots()
    ax_vel.plot(time,human_velocities, c='#F79646',label='Go Slow', lw=3)
    ax_vel.plot(time,ex_human_velocities, c='#4BACC6', label='Go Fast',lw=3)
    ax_vel.plot(time,no_human_velocities, c='#BFBFBF', label='Baseline',lw=3)
    #ax_vel.set_title('Human speed')
    ax_vel.set_xlabel('Time (seconds)')
    ax_vel.set_ylabel('Speed (m/s)')
    ax_vel.legend(loc=2)

    plt.show()


if __name__=="__main__":
    plot()