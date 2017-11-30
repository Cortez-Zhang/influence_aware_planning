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
import moveit_commander
from jaco import JacoInterface
from jaco_trajopt import CostFunction
from geometry_msgs.msg import Point, PoseStamped, Pose, Quaternion
from tf import TransformBroadcaster, TransformListener
from math import sin

class Human(InteractiveMarkerAgent):
    def __init__(self, moving_frame="moving_frame"):
        initial_position = Point(-.5,0.216,0.538)
        InteractiveMarkerAgent.__init__(self, "Human", initial_position)
        self._moving_frame = moving_frame
        self.br = TransformBroadcaster()
        self.counter = 0

        self.makeMovingMarker()
        self.server.applyChanges()
        
        #self.human_start_pose = Pose(Point(-0.5,0.216,0.538),Quaternion(0,0,0,1))
        #human_velocity = Twist(Vector3(.05,0,0),Vector3(0,0,0))
        #self.human_model = HumanModel(self.human_start_pose, human_velocity)

        #self.Timer = rospy.Timer(rospy.Duration(0.02), self._update_human_callback)
        self.start_time = rospy.Time.now().to_sec()

        self.start_human_sub = rospy.Subscriber("start_human",Pose, self._start_human)
        self.reset_human_sub = rospy.Subscriber("reset_human",Pose, self._reset_human)
        self.human_state_pub = rospy.Publisher("human_state",Pose, queue_size=10)

    def _onclick_callback(self, feedback):
        pass
    
    def _update_human_callback(self, msg):
        #rospy.loginfo("In update human callback")
        self.get_simulated_human_state()

    def get_simulated_human_state(self):
        time = rospy.Time.now()
        #keeps track of time
        time_from_start = time.to_sec() - self.start_time
        #rospy.loginfo("time_from_start {}".format(time_from_start))
        #humanpose = self.human_model.get_pose_forPlayBack(time_from_start)
        #pos = humanpose.position
        self.br.sendTransform((time_from_start*.25, 0, 0) , (0, 0, 0, 1.0), time, self._moving_frame, self._base_frame )
        #rospy.loginfo("human position {}".format(pos))
        #   self.human_state_pub.publish(Pose(pos,Quaternion(0,0,0,1)))
        #return pos
    
    def _reset_human(self, human_start_pose):
        rospy.loginfo("resetting Human")
        #self.counter = 0
        self.Timer.shutdown()
    
    def _start_human(self, human_start_pose):
        rospy.loginfo("Starting Human")
        #self.counter = 0
        #self.human_start_pose = human_start_pose
        self.start_time = rospy.Time.now().to_sec()
        self.Timer = rospy.Timer(rospy.Duration(0.02), self._update_human_callback)

    def makeMovingMarker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self._moving_frame
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