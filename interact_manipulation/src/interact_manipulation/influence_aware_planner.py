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

def main():
        jaco_interface = JacoInterface()
        jaco_interface.home()

        #get the location of the robot and use that as the start pose
        start_pose = jaco_interface.arm_group.get_current_pose()

        #create the goal_pose request for the robot
        goal_pose = PoseStamped()
        header = Header()
        header.frame_id ="root"
        header.seq = 3
        header.stamp = rospy.get_rostime()
        goal_pose.header = header
        goal_pose.pose.position = Point(-0.2,-0.2,0.538)
        goal_pose.pose.orientation = start_pose.pose.orientation
        
        #create the human model #TODO use a factory here?
        model_name = rospy.get_param("/human_model/model_name")
        goal1 = np.array([-0.4,-0.1,0.538])
        goal2 = np.array([-0.2,-0.2,0.538])
        human_start_state = HumanState(np.array([-.3,0.3,0.538]), np.array([0,0,0])) 

        if model_name == "certainty_based_speed":
            goals = [goal1, goal2]
            goal_inference = GoalInference(goals, variance = 2)
            human_model = CertaintyBasedSpeedModel(human_start_state, goals, goal_inference=goal_inference)
        elif model_name == "constant_velocity_model":
            human_model = ConstantVelocityModel(human_start_state)
        elif model_name == "potential_field_model":
            human_model = PotentialFieldModel(human_start_state, goal1)
        else:
            err = "No human model object exists with model name {}".format(model_name)
            rospy.logerr(err)
            raise ValueError(err)
        
        #create the robot cost function, including human model
        #TODO consider a Factory here so I dont have to handle all the function names
        cost_name = rospy.get_param("/cost_func/cost_name")
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

if __name__=="__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('jaco_moving_target')
    main()
    moveit_commander.roscpp_shutdown()