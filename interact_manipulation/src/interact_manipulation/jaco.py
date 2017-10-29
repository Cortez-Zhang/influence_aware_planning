#!/usr/bin/env python

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, DisplayTrajectory
from visualization_msgs.msg import Marker
from jaco_trajopt import JacoTrajopt


WORLD_FRAME = '/world'


def main():
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    display_trajectory_pub = rospy.Publisher('/move_group/display_planned_path', DisplayTrajectory, queue_size=10)

    # Set up the IK service
    rospy.wait_for_service('/compute_ik')
    compute_ik = rospy.ServiceProxy('/compute_ik', GetPositionIK)

    def ik(pose_stamped, group_name='arm'):
        """ Computes the inverse kinematics """
        req = PositionIKRequest()
        req.group_name = group_name
        req.pose_stamped = pose_stamped
        req.timeout.secs = 0.1
        req.avoid_collisions = False

        try:
            res = compute_ik(req)
            return res
        except rospy.ServiceException, e:
            print("IK service call failed: {}".format(e))

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('jaco_move_group')

    robot = moveit_commander.RobotCommander()
    print(robot.get_group_names())
    print(robot.get_current_state())

    arm_group = moveit_commander.MoveGroupCommander('arm')
    # print(arm_group.get_current_pose())

    planner = JacoTrajopt()

    # Visualize start and goal poses
    start_pose = arm_group.get_current_pose()

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
    marker_pub.publish(start_marker)

    # Construct the goal
    goal_pose = start_pose
    goal_pose.pose.position.x += 0.5
    goal_pose.pose.position.z -= 0.5

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
    marker_pub.publish(goal_marker)

    # Run IK to find the goal configuration from the goal pose
    res = ik(goal_pose)
    print(res)

    start_config = arm_group.get_current_joint_values()
    goal_config = [q for q in res.solution.joint_state.position[0:7]]
    # start_config = robot.get_current_state().joint_state.position
    # goal_config = res.solution.joint_state.position
    print("start config = {}, goal config = {}".format(start_config, goal_config))

    # Plan a trajectory with trajopt
    traj = planner.plan(start_config, goal_config)
    print(traj)

    # Display the planned trajectory
    display_traj = DisplayTrajectory()
    display_traj.trajectory_start = robot.get_current_state()
    display_traj.trajectory.append(traj)
    display_trajectory_pub.publish(display_traj)

    arm_group.execute(traj, wait=False)

    rospy.spin()

    moveit_commander.roscpp_shutdown()


if __name__ == '__main__':
    main()
