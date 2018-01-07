import rospy
import openravepy
import numpy as np
from trajopt_interface import CostFunction
import util
import marker_wrapper

import numpy as np


class AffectHumanCost(CostFunction):
    def __init__(self, robot, human_model):
        param_list = ["hit_human_penalty","normalize_sigma","care_about_distance","eef_link_name"]
        namespace = "/cost_func/"
        param_dict = util.set_params(param_list,namespace)

        CostFunction.__init__(self, params=param_dict)
        self.robot = robot #TODO replace with imported variable
        self.human_model = human_model
        self.robotDOF = rospy.get_param("/robot_description_kinematics/ndof")

    def get_cost(self, configs):
        """ Returns cost based on the distance between the end effector and the human
            this funciton is given as a callback to trajopt
            params
            ---
            configs: a list with (number of robot dof x num way points)
            given as a callback from trajopt
            Returns
            ---
            cost: A floating point cost value to be optimized by trajopt
        """
        #reshape the list of configs into the appropriate shape (DOF,N)
        configs = np.asarray(configs)
        configs = np.reshape(configs, (self.robotDOF,-1))
        
        #calculate kinematics for each configuration to find end effector in world space
        eef_positions = []
        for i in range(np.shape(configs)[1]):
            config = configs[:,i]
            eef_positions.append(self._get_OpenRaveFK(config, self.params["eef_link_name"]))
        
        # reset the human model before calculating human response - 
        # human pos will be saved for simulation
        self.human_model.reset_model()
        human_positions = self.human_model.get_human_positions(eef_positions)
        human_velocities = self.human_model.human_velocities

        return self._human_affect(human_positions,human_velocities, eef_positions)

    def _human_affect(self, human_positions, human_velocities, eef_positions):
        """
        Compute a cost that leverages some affect on the human. To be implemented by subclasses
        ---
        Params
        human_positions: list of (3,) numpy arrays
        human_velocities: list of (3,) numpy arrays representing velocity
        eef_positions: list of (3,) location of the end effector in world coordinates
        """
        pass
    
    def _get_OpenRaveFK(self, config, link_name):
        """ Calculate the forward kinematics using openRAVE for use in cost evaluation.
            Params
            ---
            config: Robot joint configuration (3,) numpy array
            link_name: Name of the link to calculate forward kinematics for
        """
        q = config.tolist()
        self.robot.SetDOFValues(q + [0.0, 0.0, 0.0])
        eef_link = self.robot.GetLink(link_name)
        if eef_link is None:
            rospy.logerror("Error: end-effector \"{}\" does not exist".format(self.params["eef_link_name"]))
            raise ValueError("Error: end-effector \"{}\" does not exist".format(self.params["eef_link_name"]))
        eef_pose = openravepy.poseFromMatrix(eef_link.GetTransform())
        return np.array([eef_pose[4], eef_pose[5], eef_pose[6]])

class HumanGoFirstCost(AffectHumanCost):
    """ Calculate the distance above midpoint for human going first
    This cost can only be used with a single goal. Its meant to be something like an intersection
    The human and robot cross paths and the robot wants the human to go first.
    """
    def _human_affect(self, human_positions, human_velocities, eef_positions):
        assert len(self.human_model.goals)==1     
        cost = 0.0
        #get midpoint between start and goal
        start_pos = self.human_model.start_state.position
        goal_pos = self.human_model.goals[0]

        midpoint = (start_pos+goal_pos)/2.0
        to_mid = midpoint-start_pos
        norm_to_mid = np.linalg.norm(to_mid)

        for position in human_positions:
            #TODO vectorize this better 
            """" Calculate individual waypoint cost """
            distance=np.dot(to_mid,midpoint-position)/norm_to_mid
            cost = np.tanh(distance)    
        return cost
     
class HumanSpeedCost(AffectHumanCost):
    """
    Penalize trajectories which make the human go fast
    """
    def _human_affect(self, human_positions, human_velocities, eef_positions):
        cost = 0.0
        for vel in human_velocities:
            speed = np.linalg.norm(vel)
            cost+=speed
        return cost

class HumanClosenessCost(AffectHumanCost):     
    """
    Penalize trajectories which are close to the human. 
    """
    def _human_affect(self, human_positions, human_velocities, eef_positions):
        cost = 0.0
        for i, (human_position, eef_position) in enumerate(zip(human_positions, eef_positions)):
            #rospy.loginfo("human_position {}".format(human_position))
            distance = np.linalg.norm(human_position - eef_position)
            if distance < self.params['care_about_distance']:
                #TODO use potential field
                # assign cost inverse proportional to the distance to human squared 
                cost += self.params['hit_human_penalty'] * 1/(distance)
        return cost