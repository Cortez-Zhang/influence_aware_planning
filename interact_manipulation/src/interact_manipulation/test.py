import numpy as np
from numpy import array

#from jaco_moving_target import HumanState, HumanModel
import matplotlib.pyplot as plt

# def human_go_first_cost(self, human_positions):
#     """Calculate the distance above midpoint for human going first
#         @Param human_positions: list of (3,) numpy arrays
#     """
#     cost = 0.0
#     #get midpoint between start and goal
#     start_pos = self.human_model.start_state.position
#     goal_pos = self.human_model.goal_pos

#     midpoint = (start_pos+goal_pos)/2.0
#     to_mid = midpoint-start_pos
#     norm_to_mid = to_mid/np.linalg.norm(to_mid)

#     for position in human_positions:
#         #TODO vectorize this better
#         distance=np.dot(to_mid,position-midpoint)/norm_to_mid
#         cost += np.tanh(distance)            
#     return cost

# human_goal_position = np.array([-.4,0.3,0.538])
# human_start_state = HumanState(np.array([-.4,0.1,0.4]), np.array([0,0,0]))
# human_model = HumanModel(human_start_state, human_goal_position, simulation_method="point_mass")

# eef_positions = [array([-0.5  ,  0.216,  0.538]), array([-0.49 ,  0.216,  0.538]), array([-0.48 ,  0.216,  0.538]), array([-0.47 ,  0.216,  0.538]), array([-0.46 ,  0.216,  0.538]), array([-0.45 ,  0.216,  0.538]), array([-0.44 ,  0.216,  0.538]), array([-0.43 ,  0.216,  0.538]), array([-0.42 ,  0.216,  0.538]), array([-0.41 ,  0.216,  0.538]), array([-0.4  ,  0.216,  0.538]), array([-0.39 ,  0.216,  0.538]), array([-0.38 ,  0.216,  0.538]), array([-0.37 ,  0.216,  0.538]), array([-0.36 ,  0.216,  0.538]), array([-0.35 ,  0.216,  0.538]), array([-0.34 ,  0.216,  0.538]), array([-0.33 ,  0.216,  0.538]), array([-0.32 ,  0.216,  0.538]), array([-0.31 ,  0.216,  0.538]), array([-0.3  ,  0.216,  0.538])]

# #plt.figure(1)
# human_model.get_human_positions(eef_positions)
# plt.figure(1)
# plt.plot(human_start_state.position[0],human_start_state.position[1], 'ro')
# plt.plot(human_goal_position[0],human_goal_position[1], 'go')

# #print("human_positions {}".format(human_model.human_positions))
# human_poses = np.asarray(human_model.human_positions)
# eef_poses = np.asarray(eef_positions)
# print(np.shape(human_poses))
# plt.plot(human_poses[:,0].T, human_poses[:,1].T, 'rx')
# plt.plot(eef_poses[:,0].T, eef_poses[:,1].T, 'gx')


# plt.figure(2)
# plt.plot(human_poses[:,1].T, 'rx')

# plt.figure(1)
# # ############################# heat map plotting code


start_pos = np.array([-1,-1,0])
goal_pos = np.array([1,1,0])

def waypoint_cost(position):
    """ Calculate individual waypoint cost """
    midpoint = (start_pos+goal_pos)/2.0
    to_mid = midpoint-start_pos

    norm_to_mid = np.linalg.norm(to_mid)
    
    distance=np.dot(to_mid,midpoint-position)/norm_to_mid
    cost = np.tanh(distance)    
    return cost

# eef_position = eef_positions[0]
# def sum_of_forces(curr_pos):
#     robot_repulsion = -1*human_model.params["robot_aggressiveness"]
#     goal_attraction = (1-human_model.params["robot_aggressiveness"])
#     F_repulse = robot_repulsion*human_model.potential_field(eef_position,curr_pos)
#     #*direction_from_robot*self.obstacle_penalty_cost(dist_from_robot)
#     F_attract = goal_attraction*human_model.potential_field(human_model.goal_pos,curr_pos)
    
#     return F_attract+F_repulse

# plt.plot(eef_position[0],eef_position[1], 'ro')

num = 100
x = np.linspace(-2,2,num)
y = np.linspace(-2,2,num)
#z = np.linspace(0.2,5,num)

XHuman, YHuman = np.meshgrid(x,y)
C = np.zeros((num,num))

ZHuman = 0
for i in range(num):
   for j in range(num):
       curr_pos = np.array([XHuman[i,j],YHuman[i,j],ZHuman ])
       C[i,j] = waypoint_cost(curr_pos)

plt.plot(start_pos[0],start_pos[1],'ko')
plt.plot(goal_pos[0], goal_pos[1], 'go')
plt.pcolor(XHuman, YHuman, C)
plt.show()