#!/usr/bin/env python

import rospkg
import json
import time
import random
import math
import numpy as np
from abc import ABCMeta, abstractmethod
import trajoptpy
import openravepy
import rospy
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray


class CostFunction:
    """ Base class for a cost function for Trajopt """

    __metaclass__ = ABCMeta

    def __init__(self, params):
        self.params = params

    @abstractmethod
    def get_cost(self, config):
        """ Returns the cost incurred at the specified configuration """
        pass


class JacoTrajopt:
    """ Interface to Trajopt planner and OpenRAVE """

    def __init__(self):
        self.env = openravepy.Environment()

        rospack = rospkg.RosPack()
        package_path = rospack.get_path('interact_manipulation')
        jaco_urdf_path = package_path + '/config/jaco.urdf'
        jaco_srdf_path = package_path + '/config/jaco.srdf'
        print("Loading Jaco URDF from {} and SRDF from {}...".format(jaco_urdf_path,
                                                                     jaco_srdf_path))

        self.urdf_module = openravepy.RaveCreateModule(self.env, 'urdf')
        name = self.urdf_module.SendCommand("load {} {}".format(jaco_urdf_path,
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

        self.trajopt_num_waypoints = 10
        self.cost_functions = []

    def load_body_from_urdf(self, path_to_urdf, transform=np.eye(4, 4)):
        """ Load a body (non-robot object) from a URDF file into the OpenRAVE environment """
        name = self.urdf_module.SendCommand("load {}".format(path_to_urdf))
        body = self.env.GetKinBody(name)
        body.SetTransform(transform)

    def get_body_markers(self):
        """ Returns a list of visualization_msgs/MarkerArray with all the links of each body in the environment """
        body_markers = []

        # Get all the bodies in the OpenRAVE environment
        bodies = self.env.GetBodies()
        for body in bodies:
            print("Found body with name: {}".format(body.GetName()))
            body_marker = MarkerArray()

            # Choose a random color for this body
            color_r = random.random()
            color_g = random.random()
            color_b = random.random()

            # Create a separate marker for each link
            for link in body.GetLinks():
                print("  Link name: {}".format(link.GetName()))
                link_transform = link.GetTransform()

                link_marker = Marker()
                link_marker.header.frame_id = '/world'
                link_marker.header.stamp = rospy.get_rostime()
                link_marker.ns = body.GetName() + '/link/' + link.GetName()
                link_marker.id = 0
                link_marker.type = Marker.SPHERE

                pose = openravepy.poseFromMatrix(link_transform)

                link_marker.pose.position.x = pose[4]
                link_marker.pose.position.y = pose[5]
                link_marker.pose.position.z = pose[6]
                link_marker.pose.orientation.x = pose[1]
                link_marker.pose.orientation.y = pose[2]
                link_marker.pose.orientation.z = pose[3]
                link_marker.pose.orientation.w = pose[0]

                link_marker.scale.x = 0.2
                link_marker.scale.y = 0.1
                link_marker.scale.z = 0.1
                link_marker.color.r = color_r
                link_marker.color.g = color_g
                link_marker.color.b = color_b
                link_marker.color.a = 0.50
                link_marker.lifetime = rospy.Duration(0)
                body_marker.markers.append(link_marker)

            body_markers.append(body_marker)

        return body_markers

    def plan(self, start_config, goal_config):
        """ Plan from a start configuration to goal configuration """
        print("Planning from config {} to {}...".format(start_config,
                                                        goal_config))

        dofs = len(start_config)

        start_config[2] += math.pi  # TODO this seems to be a bug in OpenRAVE?
        goal_config[2] += math.pi  # TODO this seems to be a bug in OpenRAVE?

        self.jaco.SetDOFValues(start_config + self.finger_joint_values)

        request = {
            "basic_info":
                {
                    "n_steps": self.trajopt_num_waypoints,
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

        # Add the cost function
        for i, cost_function in enumerate(self.cost_functions):
            for t in range(1, self.trajopt_num_waypoints):
                prob.AddCost(cost_function.get_cost, [(t, j) for j in range(dofs)], "cost_{}_waypoint_{}".format(i, t))

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
            p.positions[2] -= math.pi  # TODO this seems to be a bug in OpenRAVE?
            p.time_from_start = rospy.Duration(t)
            t += 0.5

            msg.joint_trajectory.points.append(p)

        return msg
