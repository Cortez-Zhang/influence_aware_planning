#!/usr/bin/python
import math
import openravepy
import rospy
import random
import copy

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
import moveit_commander
from jaco import JacoInterface
from jaco_trajopt import CostFunction
from geometry_msgs.msg import Point, PoseStamped, Pose
from tf import TransformBroadcaster, TransformListener
from math import sin

def normalize_exp(x, sigma):
    """ Normalizes the value using an exponential """
    return 1.0 - math.exp(-math.pow(x, 2) / sigma)

class WaypointCostFunction(CostFunction):
    def __init__(self, robot, eef_link_name='j2s7s300_end_effector'):
        CostFunction.__init__(self, params={'waypoint_1': 0.5,
                                            'waypoint_2': 0.0,
                                            'normalize_sigma': 1.0})

        self.robot = robot
        self.eef_link_name = eef_link_name
        self._care_about_distance = .1

        waypoint_1 = Pose()
        waypoint_1.position.x = 0.4
        waypoint_1.position.y = 0.4
        waypoint_1.position.z = 0.85
        waypoint_1.orientation.w = 1.0
        waypoint_2 = Pose()
        waypoint_2.position.x = 0.4
        waypoint_2.position.y = -0.4
        waypoint_2.position.z = 0.85
        waypoint_2.orientation.w = 1.0
        # self.waypoints = [waypoint_1, waypoint_2]
        self.waypoints = [waypoint_1]

    def get_cost(self, config):
        """define the cost for a given configuration"""
        q = config.tolist()
        # q[2] += math.pi  # TODO this seems to be a bug in OpenRAVE?
        self.robot.SetDOFValues(q + [0.0, 0.0, 0.0])
        eef_link = self.robot.GetLink(self.eef_link_name)
        if eef_link is None:
            print("Error: end-effector \"{}\" does not exist".format(self.eef_link_name))
            return 0.0

        eef_pose = openravepy.poseFromMatrix(eef_link.GetTransform())

        cost = 0.0
        for i, waypoint in enumerate(self.waypoints):
            # Get the (normalized) distance from the end-effector to the waypoint
            distance = normalize_exp(self._dist(eef_pose, waypoint), sigma=self.params['normalize_sigma'])
            if abs(distance) < self._care_about_distance:
                # assign cost inverse proportional to the distance squared 
                # TODO swap this with something more principled
                cost += self.params['waypoint_{}'.format(i + 1)] * 1/math.pow(distance,2)
                # c += self.params['waypoint_{}'.format(i + 1)] * self._dist(eef_pose, waypoint)

        return cost

    def _dist(self, eef_pose, waypoint):
        return math.sqrt(math.pow(eef_pose[4] - waypoint.position.x, 2) +
                         math.pow(eef_pose[5] - waypoint.position.y, 2) +
                         math.pow(eef_pose[6] - waypoint.position.z, 2))

    def get_waypoint_markers(self):
        markers = MarkerArray()

        for i, waypoint in enumerate(self.waypoints):
            # Choose a random color for this body
            color_r = random.random()
            color_g = random.random()
            color_b = random.random()

            waypoint_marker = Marker()
            waypoint_marker.header.frame_id = '/world'
            waypoint_marker.header.stamp = rospy.get_rostime()
            waypoint_marker.ns = '/waypoint'
            waypoint_marker.id = i
            waypoint_marker.type = Marker.SPHERE
            waypoint_marker.pose = waypoint
            waypoint_marker.scale.x = 0.1
            waypoint_marker.scale.y = 0.1
            waypoint_marker.scale.z = 0.1
            waypoint_marker.color.r = color_r
            waypoint_marker.color.g = color_g
            waypoint_marker.color.b = color_b
            waypoint_marker.color.a = 0.50
            waypoint_marker.lifetime = rospy.Duration(0)
            markers.markers.append(waypoint_marker)

            text_marker = Marker()
            text_marker.header.frame_id = '/world'
            text_marker.header.stamp = rospy.get_rostime()
            text_marker.ns = '/waypoint/text'
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.pose = waypoint
            text_marker.scale.z = 0.05
            text_marker.color.r = color_r
            text_marker.color.g = color_g
            text_marker.color.b = color_b
            text_marker.color.a = 0.50
            text_marker.text = 'waypoint {}'.format(i + 1)
            text_marker.lifetime = rospy.Duration(0)
            markers.markers.append(text_marker)

        return markers

class InteractiveMarkerAgent():
    def __init__(self, server_name, position, menu_labels=[], base_frame = "root", scale = .25):
        self._scale = scale
        self.server = InteractiveMarkerServer(server_name)
        self.menu_handler = MenuHandler()
        self.menu_labels = menu_labels
        self.setup_menu_handler()
        self.marker_position = Point(position.x, position.y, position.z)
        self._base_frame = base_frame

    def setup_menu_handler(self):
        for menu_label in self.menu_labels:
            self.menu_handler.insert(menu_label, callback=self._onclick_callback )

    def _onclick_callback(self, feedback):
        pass
    
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
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append( self.makeBox(msg) )
        msg.controls.append( control )
        return control

class Human(InteractiveMarkerAgent):
    def __init__(self, moving_frame="moving_frame"):
        initial_position = Point(0,0,0)
        InteractiveMarkerAgent.__init__(self, "Human", initial_position)
        self._moving_frame = moving_frame
        self.br = TransformBroadcaster()
        self.counter = 0

        self.makeMovingMarker()
        self.server.applyChanges()
        self.Timer = rospy.Timer(rospy.Duration(0.01), self._update_human_callback)
        self.Pub = rospy.Publisher('chatter', String, queue_size=10)

    def _onclick_callback(self, feedback):
        pass
    
    def _update_human_callback(self, msg):
        #rospy.loginfo("In update human callback")
        self.get_simulated_human_state()

    def get_simulated_human_state(self):
        time = rospy.Time.now()
        translation = (0, 0, sin(self.counter/140.0)*2.0)
        self.br.sendTransform(translation , (0, 0, 0, 1.0), time, self._moving_frame, self._base_frame )
        #rospy.loginfo("moving Frame: {} static frame: {} translation: {}".format(self._moving_frame, self._base_frame, translation))
        self.counter += 1
        return translation
    
    def reset_human(self):
        rospy.loginfo("resetting Human")
        self.counter = 0
        self.Timer.shutdown()
    
    def start_human(self):
        rospy.loginfo("Starting Human")
        self.counter = 0
        self.Timer = rospy.Timer(rospy.Duration(0.01), self._update_human_callback)

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

class AssertiveRobotPlanner(InteractiveMarkerAgent):
    def __init__(self):
        initial_position = Point(0,0,0)
        menu_label_list = []
        menu_label_list.append("Plan and Execute")
        menu_label_list.append("Reset Human")
        menu_label_list.append("Reset Robot")
        InteractiveMarkerAgent.__init__(self, "End_Goal", initial_position, menu_labels=menu_label_list)

        self.jaco_interface = JacoInterface()
        self.waypoint_cost_func = WaypointCostFunction(self.jaco_interface.planner.jaco)
        self.human = Human()

        self.make6DofMarker(False, InteractiveMarkerControl.MOVE_ROTATE_3D, True )
        self.server.applyChanges()

    def _onclick_callback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            if feedback.menu_entry_id == 1:
                self.human.start_human()
                self._plan_and_execute(feedback)

            elif feedback.menu_entry_id == 2:
                self.human.reset_human()
            elif feedback.menu_entry_id == 3:
                self.jaco_interface.home()

        self.server.applyChanges()

    def _plan_and_execute(self, feedback):
        self.jaco_interface.planner.cost_functions = [self.waypoint_cost_func]
        self.jaco_interface.planner.trajopt_num_waypoints = 15

        goal_pose = PoseStamped()
        goal_pose.header = feedback.header
        goal_pose.pose = feedback.pose

        start_pose = self.jaco_interface.arm_group.get_current_pose()
        rospy.loginfo("Requesting plan from start_pose:\n {} \n goal_pose:\n {}".format(start_pose, goal_pose))

        traj = self.jaco_interface.plan(start_pose, goal_pose)

        m = self.waypoint_cost_func.get_waypoint_markers()
        self.jaco_interface.marker_array_pub.publish(m)

        self.jaco_interface.execute(traj)

    def run(self):        
        self.jaco_interface.home()

        rospy.spin()

    def make6DofMarker(self, fixed, interaction_mode, show_6dof = False):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self._base_frame
        int_marker.pose.position = self.marker_position
        int_marker.scale = self._scale

        int_marker.name = "simple_6dof"
        int_marker.description = "Simple 6-DOF Control"

        # insert a box
        self.makeBoxControl(int_marker)
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

        self.server.insert(int_marker, self._onclick_callback)
        self.menu_handler.apply( self.server, int_marker.name )
    
if __name__=="__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('jaco_moving_target')
    human_aware_planner = AssertiveRobotPlanner()
    human_aware_planner.run()
    moveit_commander.roscpp_shutdown()