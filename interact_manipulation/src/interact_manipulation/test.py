import numpy as np
from numpy import array

from jaco_moving_target import *
import matplotlib.pyplot as plt

human_goal_position = np.array([0,0.216,0.538])
human_start_state = HumanState(np.array([-0.5,0.216,0.538]), np.array([.05,0,0]))
human_model = HumanModel(human_start_state, human_goal_position, simulation_method="point_mass")

eef_positons = [array([-0.5  ,  0.216,  0.538]), array([-0.49 ,  0.216,  0.538]), array([-0.48 ,  0.216,  0.538]), array([-0.47 ,  0.216,  0.538]), array([-0.46 ,  0.216,  0.538]), array([-0.45 ,  0.216,  0.538]), array([-0.44 ,  0.216,  0.538]), array([-0.43 ,  0.216,  0.538]), array([-0.42 ,  0.216,  0.538]), array([-0.41 ,  0.216,  0.538]), array([-0.4  ,  0.216,  0.538]), array([-0.39 ,  0.216,  0.538]), array([-0.38 ,  0.216,  0.538]), array([-0.37 ,  0.216,  0.538]), array([-0.36 ,  0.216,  0.538]), array([-0.35 ,  0.216,  0.538]), array([-0.34 ,  0.216,  0.538]), array([-0.33 ,  0.216,  0.538]), array([-0.32 ,  0.216,  0.538]), array([-0.31 ,  0.216,  0.538]), array([-0.3  ,  0.216,  0.538])]

human_model.get_human_positions(eef_positions)

plt.plot(human_start_state.position[0:1], 'go')
plt.plot(human_goal_position[0:1], 'ro')
plt.plot()
plt.show()

