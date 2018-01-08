#!/usr/bin/python
import openravepy
import rospy
import datetime

from geometry_msgs.msg import *
from std_msgs.msg import Empty, Float32MultiArray, MultiArrayLayout, MultiArrayDimension, Header
from visualization_msgs.msg import *

import marker_wrapper
import util

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

