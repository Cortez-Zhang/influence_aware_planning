#!/usr/bin/python
from jaco_moving_target import *
import math
import openravepy
import rospy
import random
import copy
import numpy as np

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from std_msgs.msg import Float32MultiArray
import moveit_commander
from jaco import JacoInterface
from jaco_trajopt import CostFunction
from geometry_msgs.msg import Point, PoseStamped, Pose, Quaternion
from tf import TransformBroadcaster, TransformListener
from math import sin

class Human(InteractiveMarkerAgent):
    def __init__(self):
        initial_position = Point(0,0,0) #this isnt realy the initial pos
        InteractiveMarkerAgent.__init__(self, "Human", initial_position)
        self.moving_frame = "moving_frame"
        self.br = TransformBroadcaster()

        self.makeMovingMarker()
        self.server.applyChanges()
        
        self.start_time = rospy.Time.now().to_sec()
        self.simulated_dt = 0.2 #TODO set this better, maybe use parameter server?
        self.playback_dt = 0.02 #TODO set this better

        self.playback_positions = None
        self.counter = 0

        self.start_human_sub = rospy.Subscriber("start_human", Float32MultiArray, self._start_human)
        self.reset_human_sub = rospy.Subscriber("reset_human", Empty, self._reset_human)

    def _onclick_callback(self, feedback):
        pass
    
    def set_states(self, sim_pos):
        """ Interpolates simulated_positions and sets self.playback_positions (3,num_playback_wpts) numpy array
            -------
            Params: sim_pos a (num_sim_wpts,3) numpy array
        """
        num_sim_wpts = 21 #TODO set this with a parameter server var
        end_time = num_sim_wpts*self.simulated_dt
        num_playback_wpts = end_time/self.playback_dt

        rospy.loginfo("np.shape(sim_pos)[1] {}".format(np.shape(sim_pos)[0]))
        rospy.loginfo("num_sim_wpts {}".format(num_sim_wpts))

        assert np.shape(sim_pos)[0] == num_sim_wpts

        x = np.linspace(0, end_time, num=num_sim_wpts, endpoint=True)
        y = np.linspace(0, end_time, num=num_sim_wpts, endpoint=True)
        z = np.linspace(0, end_time, num=num_sim_wpts, endpoint=True)

        xnew = np.linspace(0, end_time, num=num_playback_wpts, endpoint=True)
        ynew = np.linspace(0, end_time, num=num_playback_wpts, endpoint=True)
        znew = np.linspace(0, end_time, num=num_playback_wpts, endpoint=True)

        playback_pos = np.empty((num_playback_wpts,3))
        playback_pos[:] = np.nan
       
        playback_pos[:,0] = np.interp(xnew, x, sim_pos[:,0])
        playback_pos[:,1] = np.interp(ynew, y, sim_pos[:,1])
        playback_pos[:,2] = np.interp(znew, z, sim_pos[:,2])
        rospy.loginfo("playback_pos {}".format(playback_pos))

        self.playback_positions = playback_pos

    def _update_human_callback(self, msg):
        """ Callback used to update the position of the marker through a tf broadcaster """
        time = rospy.Time.now()
        self.counter +=1
        if self.counter < np.shape(self.playback_positions)[0]:
            x = self.playback_positions[self.counter,0]
            y = self.playback_positions[self.counter,1]
            z = self.playback_positions[self.counter,2]
            self.br.sendTransform((x, y, z) , (0, 0, 0, 1.0), time, self.moving_frame, self.base_frame )
    
    def _reset_human(self, human_start_pose):
        """ Callback used with reset_human subsciber """
        rospy.loginfo("resetting Human")
        self.Timer.shutdown()
    
    def _start_human(self, human_traj_msg):
        """ Callback used to start human """
        rospy.loginfo("Starting Human. Trajmsg: {}".format(human_traj_msg))
        self.counter = 0
        human_traj = np.asarray(human_traj_msg.data)

        #rospy.loginfo("human_traj {}".format(np.shape(human_traj)) )
        human_traj = np.reshape(human_traj,(-1,3))
        rospy.loginfo("human_trajector after reshape {}".format(human_traj))
        #print(human_traj)
        self.set_states(human_traj)
        self.start_time = rospy.Time.now().to_sec()
        self.Timer = rospy.Timer(rospy.Duration(self.playback_dt), self._update_human_callback)

    def makeMovingMarker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.moving_frame
        int_marker.pose.position = self.marker_position
        int_marker.scale = self._scale

        int_marker.name = "Human"
        int_marker.description = "Simulated"

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(copy.deepcopy(control))

        control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        control.always_visible = True
        control.markers.append( self.makeBox(int_marker) )
        int_marker.controls.append(control)

        self.server.insert(int_marker, self._onclick_callback)

def starthuman():
    rospy.init_node("Human")
    human = Human()
    rospy.spin()

if __name__ == '__main__':
    starthuman()