#!/usr/bin/env python

import rospkg
import json
import time
import trajoptpy
import openravepy
import rospy
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class JacoTrajopt:
    def __init__(self):
        self.env = openravepy.Environment()

        rospack = rospkg.RosPack()
        package_path = rospack.get_path('interact_manipulation')
        jaco_urdf_path = package_path + '/config/jaco.urdf'
        jaco_srdf_path = package_path + '/config/jaco.srdf'
        print("Loading Jaco URDF from {} and SRDF from {}...".format(jaco_urdf_path,
                                                                     jaco_srdf_path))

        urdf_module = openravepy.RaveCreateModule(self.env, 'urdf')
        name = urdf_module.SendCommand("load {} {}".format(jaco_urdf_path,
                                                           jaco_srdf_path))
        self.jaco = self.env.GetRobot(name)
        self.finger_joint_values = [0.0, 0.0, 0.0]
        self.joint_names = ['j2s7s300_joint_1',
                            'j2s7s300_joint_2',
                            'j2s7s300_joint_3',
                            'j2s7s300_joint_4',
                            'j2s7s300_joint_5',
                            'j2s7s300_joint_6',
                            'j2s7s300_joint_7']

    def plan(self, start_config, goal_config):
        """ Plan from a start configuration to goal configuration """
        print("Planning from config {} to {}...".format(start_config,
                                                        goal_config))
        # print(self.jaco.GetDOFValues())
        self.jaco.SetDOFValues(start_config + self.finger_joint_values)
        request = {
            "basic_info":
                {
                    "n_steps": 10,
                    "manip": self.jaco.GetActiveManipulator().GetName(),
                    "start_fixed": True
                },
            "costs":
                [
                    {
                        "type": "joint_vel",  # joint-space velocity cost
                        "params": {"coeffs": [1]} # a list of length one is automatically expanded to a list of length n_dofs
                    }
                ],
            "constraints":
                [
                    {
                        "type": "joint",  # joint-space target
                        "params": {"vals": goal_config}  # length of vals = # dofs of manip
                    }
                ],
            "init_info": {
                "type": "straight_line",  # straight line in joint space.
                "endpoint": goal_config
            }
        }
        s = json.dumps(request)  # convert dictionary into json-formatted string
        prob = trajoptpy.ConstructProblem(s, self.env)  # create object that stores optimization problem
        t_start = time.time()
        result = trajoptpy.OptimizeProblem(prob)  # do optimization
        t_elapsed = time.time() - t_start
        print("Planning took {} seconds".format(t_elapsed))
        print(result)
        return self._to_trajectory_msg(result.GetTraj())

    def _to_trajectory_msg(self, traj):
        """ Converts to a moveit_msgs/RobotTrajectory message """
        msg = RobotTrajectory()
        msg.joint_trajectory.joint_names = self.joint_names
        t = 0.0
        for i in range(traj.shape[0]):
            p = JointTrajectoryPoint()
            p.positions = traj[i, :].tolist()
            p.time_from_start = rospy.Duration(t)
            t += 0.5

            msg.joint_trajectory.points.append(p)

        return msg