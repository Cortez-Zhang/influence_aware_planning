import numpy as np
from numpy import array

from jaco_moving_target import HumanState, HumanModel
import matplotlib.pyplot as plt

human_goal_position = np.array([-.4,0.3,0.538])
human_start_state = HumanState(np.array([-.4,0.1,0.4]), np.array([0,0,0]))
human_model = HumanModel(human_start_state, human_goal_position, simulation_method="point_mass")

eef_positions = [array([-0.5  ,  0.216,  0.538]), array([-0.49 ,  0.216,  0.538]), array([-0.48 ,  0.216,  0.538]), array([-0.47 ,  0.216,  0.538]), array([-0.46 ,  0.216,  0.538]), array([-0.45 ,  0.216,  0.538]), array([-0.44 ,  0.216,  0.538]), array([-0.43 ,  0.216,  0.538]), array([-0.42 ,  0.216,  0.538]), array([-0.41 ,  0.216,  0.538]), array([-0.4  ,  0.216,  0.538]), array([-0.39 ,  0.216,  0.538]), array([-0.38 ,  0.216,  0.538]), array([-0.37 ,  0.216,  0.538]), array([-0.36 ,  0.216,  0.538]), array([-0.35 ,  0.216,  0.538]), array([-0.34 ,  0.216,  0.538]), array([-0.33 ,  0.216,  0.538]), array([-0.32 ,  0.216,  0.538]), array([-0.31 ,  0.216,  0.538]), array([-0.3  ,  0.216,  0.538])]

#plt.figure(1)
human_model.get_human_positions(eef_positions)
plt.figure(1)
plt.plot(human_start_state.position[0],human_start_state.position[1], 'ro')
plt.plot(human_goal_position[0],human_goal_position[1], 'go')

#print("human_positions {}".format(human_model.human_positions))
human_poses = np.asarray(human_model.human_positions)
eef_poses = np.asarray(eef_positions)
print(np.shape(human_poses))
plt.plot(human_poses[:,0].T, human_poses[:,1].T, 'rx')
plt.plot(eef_poses[:,0].T, eef_poses[:,1].T, 'gx')


plt.figure(2)
plt.plot(human_poses[:,0].T, 'rx')

plt.figure(1)
# ############################# heat map plotting code
eef_position = eef_positions[0]
def sum_of_forces(curr_pos):
    robot_repulsion = human_model.params["robot_repulsion"]
    goal_attraction = human_model.params["goal_attraction"]
    F_repulse = robot_repulsion*human_model.potential_field(eef_position,curr_pos)
    #*direction_from_robot*self.obstacle_penalty_cost(dist_from_robot)
    F_attract = goal_attraction*human_model.potential_field(human_model.goal_pos,curr_pos)
    
    return F_attract+F_repulse

plt.plot(eef_position[0],eef_position[1], 'ro')

num = 100
x = np.linspace(-1,1,num)
y = np.linspace(-1,1,num)
#z = np.linspace(0.2,5,num)

XHuman, YHuman = np.meshgrid(x,y)
F = np.zeros((num,num))

ZHuman = 0.538
for i in range(num):
   for j in range(num):
       curr_pos = np.array([XHuman[i,j],YHuman[i,j],ZHuman ])
       F[i,j] = np.linalg.norm(sum_of_forces(curr_pos))
       
plt.pcolor(XHuman, YHuman, F)
plt.show()