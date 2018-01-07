import rospy
import openravepy
import numpy as np
from trajopt_interface import CostFunction
import util
import marker_wrapper


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
        return cost/2.0

class SimplePointSimulator(object):
    """ A simple simulator to show markers for a human and robot """
    def __init__(self, robot_positions, human_positions, jaco_interface=None, repeat=True):
        self.Timer = None
        self.repeat = repeat #whether or not the code should run continuously

        self.simulated_dt = rospy.get_param("/human_model/dt")
        self.playback_dt = 0.02

        self.counter = 0

        self.jaco_interface = jaco_interface
        human_positions = self.process_positionlist(human_positions[0:-1])
        self.human_positions = human_positions
        
        if jaco_interface:
            robot_positions = self.process_trajmsg(robot_positions)
        else:
            robot_positions = self.process_positionlist(robot_positions)
        
        self.robot_positions = robot_positions
        assert len(self.human_positions)==len(self.robot_positions)
        
    def simulate(self):
        """Start a simulator to simulate the playback"""
        if not self.Timer:
            self.Timer = rospy.Timer(rospy.Duration(self.playback_dt), self._advance_models_callback)
  
    def _advance_models_callback(self,msg):
        """Callback for simulation"""
        if self.counter < len(self.human_positions):
            human_pos = self.human_positions[self.counter]
            robot_pos = self.robot_positions[self.counter]

            marker_wrapper.show_position_marker(human_pos, label="human\n\n", ident = 4)
            marker_wrapper.show_position_marker(robot_pos, label="robot\n\n", ident = 5)
            self.counter +=1

        else:
            self.counter = 0
            if not self.repeat:
                self.Timer.shutdown
                self.Timer = None

    def process_trajmsg(self, robot_positions):
        """ Processes robot positions into (num_sim_points,3) numpy array
                @Param: robot_positions a trajectory_msgs/JointTrajectoryPoint[] points array
                @Return: an interpolated () numpy array representing locations for simulation
        """
        rospy.loginfo("processing robot_positions****")
        processed_positions = []
        for point in robot_positions:
            joints = point.positions
            pose = self.jaco_interface.fk(joints)
            pos = pose.pose_stamped[0].pose.position
            position = np.array([pos.x,pos.y,pos.z])
            processed_positions.append(position)
        
        return self.interpolate_positions(np.asarray(processed_positions))

    def process_positionlist(self, human_positions):
        """ Processes human_positions into a (N,3) numpy array
            @Param: human_positions a list of 
        """
        rospy.loginfo("processing human_positions {}".format(human_positions))
        human_pos = np.asarray(human_positions)
        return self.interpolate_positions(human_pos)
    
    def interpolate_positions(self, sim_pos):
        """ Interpolates simulated_positions and sets self.playback_positions (3,num_playback_wpts)
            numpy array
            @Param sim_pos: A (num_sim_wpts,3) numpy array
        """
        num_sim_wpts = rospy.get_param("/low_level_planner/num_waypoints")
        end_time = num_sim_wpts*self.simulated_dt
        num_playback_wpts = end_time/self.playback_dt

        rospy.loginfo("np.shape(sim_pos)[1] {}".format(np.shape(sim_pos)[0]))
        rospy.loginfo("num_sim_wpts {}".format(num_sim_wpts))

        assert np.shape(sim_pos)[0] == num_sim_wpts

        x = np.linspace(0, end_time, num=num_sim_wpts, endpoint=True)
        xnew = np.linspace(0, end_time, num=num_playback_wpts, endpoint=True)

        playback_pos = np.empty((num_playback_wpts,3))
        playback_pos[:] = np.nan

        rospy.loginfo("sim_pos shape: {}".format(np.shape(sim_pos)))
        for idx in range(3):
            playback_pos[:,idx] = np.interp(xnew, x, sim_pos[:,idx])

        rospy.loginfo("playback_pos {}".format(playback_pos))

        return playback_pos
