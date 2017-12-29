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
import angles
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray


class CostFunction:
    """ Base class for a cost function for Trajopt """

    __metaclass__ = ABCMeta

    METHOD_NUMERICAL = "numerical"
    METHOD_ANALYTIC = "analytic"

    def __init__(self, theta, type="ABS", diffmethod=METHOD_NUMERICAL):
        self.theta = theta
        self.type = type
        self.diffmethod = diffmethod

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_visualization_markers(self):
        pass

    @abstractmethod
    def get_feature(self, config):
        """ Returns the feature vector of the robot at this configuration """
        pass

    @abstractmethod
    def get_feature_jacobian(self, config):
        pass

    def get_cost(self, config):
        """ Returns the cost incurred at the specified configuration """
        phi = self.get_feature(config)
        return self.theta * phi

    def get_cost_jacobian(self, config):
        phi_jacobian = self.get_feature_jacobian(config)
        return self.theta * phi_jacobian


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
        # self.finger_joint_values = [0.0, 0.0, 0.0]
        self.finger_joint_values = [1.0, 1.0, 1.0]
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
        self.env.Add(body, True)

    def add_cube(self, x, y, z, dim_x, dim_y, dim_z, name='cube'):
        body = openravepy.RaveCreateKinBody(self.env, '')
        body.InitFromBoxes(np.array([[0.0, 0.0, 0.0, dim_x, dim_y, dim_z]]))
        body.SetTransform([[1.0, 0.0, 0.0, x],
                           [0.0, 1.0, 0.0, y],
                           [0.0, 0.0, 1.0, z],
                           [0.0, 0.0, 0.0, 1.0]])
        body.SetName(name)
        self.env.Add(body, True)

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

        start_config_or = list(start_config)
        goal_config_or = list(goal_config)
        start_config_or[2] += math.pi  # TODO this seems to be a bug in OpenRAVE?
        goal_config_or[2] += math.pi  # TODO this seems to be a bug in OpenRAVE?

        self.jaco.SetDOFValues(start_config_or + self.finger_joint_values)

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
                    },
                    {
                        "type": "collision",
                        "params": {
                            # "coeffs": [20],
                        "coeffs": [100.0],
                        # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
                            "dist_pen": [0.025],
                        # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
                            "first_step": 0,
                            "last_step": self.trajopt_num_waypoints - 1,
                            "continuous": True
                        },
                    }
                ],
            "constraints":
                [
                    {
                        "type": "joint",  # joint-space target
                        "params": {"vals": goal_config_or}  # length of vals = # dofs of manip
                    }
                ],
            "init_info": {
                "type": "straight_line",  # straight line in joint space.
                "endpoint": goal_config_or
            }
        }
        s = json.dumps(request)  # convert dictionary into json-formatted string
        prob = trajoptpy.ConstructProblem(s, self.env)  # create object that stores optimization problem

        # Add the cost function
        for i, cost_function in enumerate(self.cost_functions):
            for t in range(1, self.trajopt_num_waypoints):
                # prob.AddCost(cost_function.get_cost, [(t, j) for j in range(dofs)], cost_function.get_name())
                if cost_function.diffmethod == CostFunction.METHOD_NUMERICAL:
                    prob.AddErrorCost(cost_function.get_cost,
                                      [(t, j) for j in xrange(dofs)],
                                      cost_function.type,
                                      "{}_{}".format(cost_function.get_name(), t))
                elif cost_function.diffmethod == CostFunction.METHOD_ANALYTIC:
                    prob.AddErrorCost(cost_function.get_cost,
                                      cost_function.get_cost_jacobian,
                                      [(t, j) for j in xrange(dofs)],
                                      cost_function.type,
                                      "{}_{}".format(cost_function.get_name(), t))

        t_start = time.time()
        result = trajoptpy.OptimizeProblem(prob)  # do optimization
        t_elapsed = time.time() - t_start
        print("Planning took {} seconds".format(t_elapsed))
        print(result.GetCosts())
        return self._to_trajectory_msg(result.GetTraj())

    def plan_pose(self, start_config, goal_pose, orientation=True):
        """
        Plans from a start configuration to a goal pose (with or without taking orientation as a goal constraint)
        
        :param start_config: the start pose
        :param goal_pose: the goal pose
        :param orientation: if True, takes orientation as a constraint at the goal pose 
        :return: 
        """
        print("Planning from config {} to pose {}...".format(start_config, goal_pose))

        dofs = len(start_config)

        start_config_or = list(start_config)
        start_config_or[2] += math.pi  # TODO this seems to be a bug in OpenRAVE?

        self.jaco.SetDOFValues(start_config_or + self.finger_joint_values)

        if orientation:
            rot_coeffs = [10, 10, 10]
        else:
            rot_coeffs = [0, 0, 0]

        constraints = [{
            "type": "pose",
            "params": {"xyz": [goal_pose.position.x,
                               goal_pose.position.y,
                               goal_pose.position.z],
                       "wxyz": [1,0,0,0],  # unused
                       "link": "j2s7s300_end_effector",
                       "rot_coeffs": rot_coeffs,
                       "pos_coeffs": [10, 10, 10]
                       }
        }]

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
                    },
                    {
                        "type": "collision",
                        "params": {
                            # "coeffs": [20],
                        "coeffs": [100.0],
                        # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
                            "dist_pen": [0.025],
                        # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
                            "first_step": 0,
                            "last_step": self.trajopt_num_waypoints - 1,
                            "continuous": True
                        },
                    }
                ],
            "constraints": constraints,
            "init_info": {
                "type": "stationary"
            }
        }
        s = json.dumps(request)  # convert dictionary into json-formatted string
        prob = trajoptpy.ConstructProblem(s, self.env)  # create object that stores optimization problem

        # Add the cost function
        for i, cost_function in enumerate(self.cost_functions):
            for t in range(1, self.trajopt_num_waypoints):
                # prob.AddCost(cost_function.get_cost, [(t, j) for j in range(dofs)], cost_function.get_name())
                if cost_function.diffmethod == CostFunction.METHOD_NUMERICAL:
                    prob.AddErrorCost(cost_function.get_cost,
                                      [(t, j) for j in xrange(dofs)],
                                      cost_function.type,
                                      "{}_{}".format(cost_function.get_name(), t))
                elif cost_function.diffmethod == CostFunction.METHOD_ANALYTIC:
                    prob.AddErrorCost(cost_function.get_cost,
                                      cost_function.get_cost_jacobian,
                                      [(t, j) for j in xrange(dofs)],
                                      cost_function.type,
                                      "{}_{}".format(cost_function.get_name(), t))

        t_start = time.time()
        result = trajoptpy.OptimizeProblem(prob)  # do optimization
        t_elapsed = time.time() - t_start
        print("Planning took {} seconds".format(t_elapsed))
        print(result.GetCosts())
        return self._to_trajectory_msg(result.GetTraj())

    def _to_trajectory_msg(self, traj, max_joint_vel=0.1):
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

        self._assign_constant_velocity_profile(msg, max_joint_vel)
        print(msg.joint_trajectory)

        return msg

    def _assign_constant_velocity_profile(self, traj, max_joint_vel):
        """ Assigns a constant velocity profile to a moveit_msgs/RobotTrajectory """
        t = 0.0
        for i in range(1, len(traj.joint_trajectory.points)):
            p_prev = traj.joint_trajectory.points[i - 1]
            p = traj.joint_trajectory.points[i]

            num_dof = len(p_prev.positions)

            max_joint_dist = 0.0
            for j in range(num_dof):
                dist = math.fabs(angles.shortest_angular_distance(p_prev.positions[j],
                                                                  p.positions[j]))
                max_joint_dist = max(max_joint_dist, dist)

            dt = max_joint_dist / max_joint_vel

            p.velocities = num_dof * [0.0]
            for j in range(num_dof):
                dist = math.fabs(angles.shortest_angular_distance(p_prev.positions[j],
                                                                  p.positions[j]))
                p.velocities[j] = dist / dt

            t += dt
            p.time_from_start = rospy.Duration(t)

        # Assign accelerations
        traj.joint_trajectory.points[0].velocities = num_dof * [0.0]
        traj.joint_trajectory.points[-1].velocities = num_dof * [0.0]
        for i in range(len(traj.joint_trajectory.points) - 1):
            p = traj.joint_trajectory.points[i]
            p_next = traj.joint_trajectory.points[i + 1]
            dt = (p_next.time_from_start - p.time_from_start).to_sec()

            p.accelerations = num_dof * [0.0]
            for j in range(num_dof):
                dv = p_next.velocities[j] - p.velocities[j]
                p.accelerations[j] = dv / dt