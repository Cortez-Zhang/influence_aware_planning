#!/usr/bin/env python

import sys
import rospy
import rospkg
import numpy as np
import moveit_commander
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from moveit_msgs.msg import PositionIKRequest, DisplayTrajectory, RobotState
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from trajopt_interface import jaco_trajopt

class JacoInterface:
    def __init__(self):
        self.robot = moveit_commander.RobotCommander()

        rospy.sleep(5)
        self.arm_group = moveit_commander.MoveGroupCommander('arm')

        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        self.marker_array_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
        self.display_trajectory_pub = rospy.Publisher('/move_group/display_planned_path',
                                                      DisplayTrajectory, queue_size=10)

        # Set up the IK service
        rospy.wait_for_service('/compute_ik')
        self.compute_ik = rospy.ServiceProxy('/compute_ik', GetPositionIK)

        # Set up the FK service
        rospy.wait_for_service('/compute_fk')
        self.compute_fk = rospy.ServiceProxy('/compute_fk', GetPositionFK)

        self.home_pose = [0.0,2.9,0.0,1.3,4.2,1.4,0.0]

    def ik(self, pose_stamped, group_name='arm'):
        """ Computes the inverse kinematics """
        req = PositionIKRequest()
        req.group_name = group_name
        req.pose_stamped = pose_stamped
        req.timeout.secs = 0.1
        req.avoid_collisions = False

        try:
            res = self.compute_ik(req)
            return res
        except rospy.ServiceException, e:
            print("IK service call failed: {}".format(e))

    def fk(self, joints, links=['j2s7s300_end_effector']):
        """ Computes the forward kinematics """
        header = Header()
        robot_state = RobotState()
        robot_state.joint_state.name = jaco_trajopt.joint_names
        robot_state.joint_state.position = joints
        return self.compute_fk(header, links, robot_state)

    def plan(self, start_pose, goal_pose):
        """ Plan from the start pose to the goal pose """
        #markers.add_position_marker(pose = start_pose.pose, label = , color = (1,0,0))
       
        # Visualize the start pose
        start_marker = Marker()
        start_marker.header.frame_id = start_pose.header.frame_id
        start_marker.header.stamp = rospy.get_rostime()
        start_marker.ns = 'planning_start'
        start_marker.id = 0
        start_marker.type = Marker.SPHERE
        start_marker.pose = start_pose.pose
        start_marker.scale.x = 0.05
        start_marker.scale.y = 0.05
        start_marker.scale.z = 0.05
        start_marker.color.r = 1.0
        start_marker.color.g = 0.0
        start_marker.color.b = 0.0
        start_marker.color.a = 0.75
        start_marker.lifetime = rospy.Duration(0)
        self.marker_pub.publish(start_marker)

        # Visualize the goal pose
        goal_marker = Marker()
        goal_marker.header.frame_id = goal_pose.header.frame_id
        goal_marker.header.stamp = rospy.get_rostime()
        goal_marker.ns = 'planning_goal'
        goal_marker.id = 0
        goal_marker.type = Marker.SPHERE
        goal_marker.pose = goal_pose.pose
        goal_marker.scale.x = 0.05
        goal_marker.scale.y = 0.05
        goal_marker.scale.z = 0.05
        goal_marker.color.r = 0.0
        goal_marker.color.g = 1.0
        goal_marker.color.b = 0.0
        goal_marker.color.a = 0.75
        goal_marker.lifetime = rospy.Duration(0)
        self.marker_pub.publish(goal_marker)

        # Run IK to find the start and goal configuration from the goal pose
        res = self.ik(goal_pose)
        goal_config = [q for q in res.solution.joint_state.position[0:7]]
        if not goal_config:
            rospy.logerr("Inverse kinematics returned empty, No solution foudn")
            traj = None
            
        else:
            res = self.ik(start_pose)
            start_config = [q for q in res.solution.joint_state.position[0:7]]

            rospy.loginfo("Planning trajopt path from start {},to goal {}".format(start_config, goal_config))
            # Plan a trajectory with trajopt
            traj = jaco_trajopt.plan(start_config, goal_config)
        #print(traj)

        return traj

    def plan_configs(self, start_config, goal_config):
        """ Plan from the start configuration to the goal configuration """

        res = self.fk(start_config, links=['j2s7s300_end_effector'])
        start_pose = res.pose_stamped[0]

        res = self.fk(goal_config, links=['j2s7s300_end_effector'])
        goal_pose = res.pose_stamped[0]

        # Visualize the start pose
        start_marker = Marker()
        start_marker.header.frame_id = start_pose.header.frame_id
        start_marker.header.stamp = rospy.get_rostime()
        start_marker.ns = 'planning_start'
        start_marker.id = 0
        start_marker.type = Marker.SPHERE
        start_marker.pose = start_pose.pose
        start_marker.scale.x = 0.1
        start_marker.scale.y = 0.1
        start_marker.scale.z = 0.1
        start_marker.color.r = 1.0
        start_marker.color.g = 0.0
        start_marker.color.b = 0.0
        start_marker.color.a = 0.75
        start_marker.lifetime = rospy.Duration(0)
        self.marker_pub.publish(start_marker)

        # Visualize the goal pose
        goal_marker = Marker()
        goal_marker.header.frame_id = goal_pose.header.frame_id
        goal_marker.header.stamp = rospy.get_rostime()
        goal_marker.ns = 'planning_goal'
        goal_marker.id = 0
        goal_marker.type = Marker.SPHERE
        goal_marker.pose = goal_pose.pose
        goal_marker.scale.x = 0.1
        goal_marker.scale.y = 0.1
        goal_marker.scale.z = 0.1
        goal_marker.color.r = 0.0
        goal_marker.color.g = 1.0
        goal_marker.color.b = 0.0
        goal_marker.color.a = 0.75
        goal_marker.lifetime = rospy.Duration(0)
        self.marker_pub.publish(goal_marker)

        print("start config = {}, goal config = {}".format(start_config, goal_config))

        # Plan a trajectory with trajopt
        traj = jaco_trajopt.plan(start_config, goal_config)
        # print(traj)

        return traj

    def execute(self, traj, wait=True, display=True):
        """ Execute the trajectory on the robot arm """
        if traj:
            rospy.loginfo("Executing trajectory")
            if display:
                # Display the planned trajectory
                display_traj = DisplayTrajectory()
                display_traj.trajectory_start = self.robot.get_current_state()
                display_traj.trajectory.append(traj)
                self.display_trajectory_pub.publish(display_traj)

            # Execute the trajectory
            self.arm_group.execute(traj, wait=wait)

    def home(self):
        initial_joints = self.home_pose
        res = self.fk(initial_joints, links=['j2s7s300_end_effector'])
        start_pose = self.arm_group.get_current_pose()
        goal_pose = res.pose_stamped[0]
        print("home pose is {}".format(goal_pose))
        traj = self.plan(start_pose, goal_pose)
        self.execute(traj)

jaco_interface = JacoInterface()

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('jaco_move_group')


    res = jaco.ik(jaco.arm_group.get_current_pose())
    start_config = [q for q in res.solution.joint_state.position[0:7]]
    start_config[2] += 3.1415
    jaco_trajopt.jaco.SetDOFValues(start_config + jaco_trajopt.finger_joint_values)

    markers = jaco_trajopt.get_body_markers()
    for m in markers:
        jaco.marker_array_pub.publish(m)

    rospy.spin()

    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
