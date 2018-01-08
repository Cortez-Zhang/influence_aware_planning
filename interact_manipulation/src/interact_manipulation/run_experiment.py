#!/usr/bin/python
import rospy

import moveit_commander
from jaco import JacoInterface

from geometry_msgs.msg import *
from std_msgs.msg import Empty, Float32MultiArray, MultiArrayLayout, MultiArrayDimension, Header
from visualization_msgs.msg import *

import marker_wrapper

from cost_affect_human import *
from human_model import *
from simulation import SimplePointSimulator

class Experiment:
    """Creates and runs an experiment"""
    def __init__(self, robot_goal_position, model_name, cost_name, jaco_interface):
        """
        Params
        ---
        robot_goal_position: A point object with the end position of the robot.
        model_name: the name of the human model to be used
        cost_name: the name of the cost function
        jaco_interface: the interface to trajopt
        """
        self.pose_start = jaco_interface.arm_group.get_current_pose()
        self.pose_goal = _get_robot_goalpose(robot_goal_position, self.start_pose.pose.orientation)
        self.human_model = ModelFactory.factory(model_name, human_start_state)
        self.cost_func = CostFunctionFactory.factory(cost_name,self.human_model, jaco_interface)
        self.jaco_interface = jaco_interface

    def _home_robot(self):
        self.jaco_interface.home()
    
    def run(self):
        #home the robot
        self._home_robot()

        #set the cost function as a callback to trajopt
        self.jaco_interface.planner.cost_functions = [self.cost_func]
        self.jaco_interface.planner.trajopt_num_waypoints = rospy.get_param("/low_level_planner/num_waypoints")
        
        rospy.loginfo("Requesting plan from start_pose:\n {} \n goal_pose:\n {}".format(start_pose, goal_pose))

        traj = self.jaco_interface.plan(start_pose, goal_pose)

        robot_positions = traj.joint_trajectory.points
        human_positions = human_model.human_positions
        
        marker_wrapper.show_position_marker(human_model.human_positions[-1], label="human end", ident=3)
        
        #create and run the simulator
        simulator = SimplePointSimulator(robot_positions, human_positions, self.jaco_interface)
        simulator.simulate()
        rospy.spin()
    
    def _get_robot_goalpose(self, position, orientation):
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

def main():
        jaco_interface = JacoInterface()
        model_name = rospy.get_param("/human_model/model_name")
        cost_name = rospy.get_param("/cost_func/cost_name")
        human_start_state = HumanState(np.array([-.3,0.3,0.538]), np.array([0,0,0])) 
        robot_goal_position = Point(-0.2,-0.2,0.538)
        
        experiment = Experiment(robot_goal_position, model_name, cost_name, jaco_interface)
        experiment.run()

if __name__=="__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('jaco_moving_target')
    main()
    moveit_commander.roscpp_shutdown()