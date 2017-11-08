#!/usr/bin/python

import rospy
from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
import moveit_commander
from jaco import JacoInterface
from geometry_msgs.msg import Point, PoseStamped

class MovingGoalPlanner():
    def __init__:
        self.server = InteractiveMarkerServer("basic_controls")
        self.menu_handler = MenuHandler()
        self.jaco = JacoInterface()
        self.menu_handler.insert( "Plan and Execute", callback=processFeedback )
        self.markerPosition = Point( 0.5, 0.5, 0.5)

    def processFeedback(self, feedback):
        print("In processFeedback")
        goal_pose = PoseStamped()
        goal_pose.header = feedback.header
        goal_pose.pose = feedback.pose

        # Move the arm to the initial configuration
        start_pose = jaco.arm_group.get_current_pose()
        #print("starting pose {}").format(start_pose)
        if feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            print("goal pose {}").format(goal_pose)

            traj = jaco.plan(start_pose, goal_pose)
            print("planned Trajectory {}").format(traj)
            jaco.execute(traj)
        server.applyChanges()

    def makeBox(self, msg):
        marker = Marker()
        marker.type = Marker.CUBE
        marker.scale.x = msg.scale * 0.45
        marker.scale.y = msg.scale * 0.45
        marker.scale.z = msg.scale * 0.45
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 1.0

        return marker

    def makeBoxControl(self, msg):

        control =  InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append( makeBox(msg) )
        msg.controls.append( control )
        return control
    # makes a marker which dynamically moves around
    def make6DofMarker(self, fixed, interaction_mode, show_6dof = False):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "root"
        int_marker.pose.position = self.markerPosition
        int_marker.scale = 1

        int_marker.name = "simple_6dof"
        int_marker.description = "Simple 6-DOF Control"

        # insert a box
        makeBoxControl(int_marker)
        int_marker.controls[0].interaction_mode = interaction_mode

        if fixed:
            int_marker.name += "_fixed"
            int_marker.description += "\n(fixed orientation)"

        if interaction_mode != InteractiveMarkerControl.NONE:
            control_modes_dict = { 
                            InteractiveMarkerControl.MOVE_3D : "MOVE_3D",
                            InteractiveMarkerControl.ROTATE_3D : "ROTATE_3D",
                            InteractiveMarkerControl.MOVE_ROTATE_3D : "MOVE_ROTATE_3D" }
            int_marker.name += "_" + control_modes_dict[interaction_mode]
            int_marker.description = "3D Control"
            if show_6dof: 
            int_marker.description += " + 6-DOF controls"
            int_marker.description += "\n" + control_modes_dict[interaction_mode]
        
        if show_6dof: 
            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 1
            control.orientation.y = 0
            control.orientation.z = 0
            control.name = "rotate_x"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 1
            control.orientation.y = 0
            control.orientation.z = 0
            control.name = "move_x"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 1
            control.orientation.z = 0
            control.name = "rotate_z"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 1
            control.orientation.z = 0
            control.name = "move_z"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 0
            control.orientation.z = 1
            control.name = "rotate_y"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 0
            control.orientation.z = 1
            control.name = "move_y"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

        server.insert(int_marker, processFeedback)
        menu_handler.apply( server, int_marker.name )

if __name__=="__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('jaco_moving_target')

    movingGoalplanner = MovingGoalPlanner()

    movingGoalplanner.make6DofMarker( False, InteractiveMarkerControl.MOVE_ROTATE_3D, position, True )

    movingGoalPlanner.jaco.home()
    
    server.applyChanges()

    rospy.spin()
    moveit_commander.roscpp_shutdown()