#!/usr/bin/python
import openravepy
import rospy
import datetime

import util

import moveit_commander
from jaco import JacoInterface

from geometry_msgs.msg import *
from std_msgs.msg import Empty, Float32MultiArray, MultiArrayLayout, MultiArrayDimension, Header
from visualization_msgs.msg import *

import marker_wrapper

from cost_affect_human import *
from human_model import *

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

class ModelFactory(object):
    @staticmethod
    def factory(model_name, human_start_state):
        goal1 = np.array([-0.4,-0.1,0.538])
        goal2 = np.array([-0.2,-0.2,0.538])

        if model_name == "certainty_based_speed":
            goals = [goal1, goal2]
            goal_inference = GoalInference(goals)
            human_model = CertaintyBasedSpeedModel(human_start_state, goals, goal_inference=goal_inference)
        elif model_name == "constant_velocity_model":
            human_model = ConstantVelocityModel(human_start_state)
        elif model_name == "potential_field_model":
            human_model = PotentialFieldModel(human_start_state, goal1)
        else:
            err = "No human model object exists with model name {}".format(model_name)
            rospy.logerr(err)
            raise ValueError(err)
        return human_model

class CostFunctionFactory(object):
    @staticmethod
    def factory(cost_name, human_model,jaco_interface):
        #TODO get rid of jaco_interface
        if cost_name == "human_speed":
            cost_func = HumanSpeedCost(jaco_interface.planner.jaco, human_model)
        elif cost_name == "human_go_first":
            cost_func = HumanGoFirstCost(jaco_interface.planner.jaco, human_model)
        elif cost_name == "human_closeness_cost":
            cost_func = HumanClosenessCost(jaco_interface.planner.jaco, human_model)
        else:
            err = "No cost object exists with cost name {}".format(cost_name)
            rospy.logerr(err)
            raise ValueError(err)
        return cost_func

def main():
        jaco_interface = JacoInterface()
        jaco_interface.home()

        #get the location of the robot and use that as the start pose
        start_pose = jaco_interface.arm_group.get_current_pose()

        #create the goal_pose request for the robot
        goal_pose = _get_robot_goalpose(Point(-0.2,-0.2,0.538), start_pose.pose.orientation)
        
        #create the human model
        model_name = rospy.get_param("/human_model/model_name")
        human_start_state = HumanState(np.array([-.3,0.3,0.538]), np.array([0,0,0])) 
        human_model = ModelFactory.factory(model_name, human_start_state)

        #create the robot cost function, including human model
        cost_name = rospy.get_param("/cost_func/cost_name")
        cost_func = CostFunctionFactory.factory(cost_name,human_model,jaco_interface)

        jaco_interface.planner.cost_functions = [cost_func]
        jaco_interface.planner.trajopt_num_waypoints = rospy.get_param("/low_level_planner/num_waypoints")
        
        rospy.loginfo("Requesting plan from start_pose:\n {} \n goal_pose:\n {}".format(start_pose, goal_pose))

        #call trajopt
        traj = jaco_interface.plan(start_pose, goal_pose)

        marker_wrapper.show_position_marker(human_model.human_positions[-1], label="human end", ident=3)

        robot_positions = traj.joint_trajectory.points
        human_positions = human_model.human_positions
        simulator = SimplePointSimulator(robot_positions, human_positions, jaco_interface)
        simulator.simulate()
        rospy.spin()

def _get_robot_goalpose(position, orientation):
    #create the goal_pose request for the robot
    goal_pose = PoseStamped()
    header = Header()
    header.frame_id ="root"
    header.seq = 3
    header.stamp = rospy.get_rostime()
    
    goal_pose.header = header
    goal_pose.pose.position = position
    goal_pose.pose.orientation = orientation
    return goal_pose

if __name__=="__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('jaco_moving_target')
    main()
    moveit_commander.roscpp_shutdown()