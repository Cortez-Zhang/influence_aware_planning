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
from geometry_msgs.msg import Point, PoseStamped, Pose, Quaternion
from tf import TransformBroadcaster, TransformListener
from math import sin
from std_msgs.msg import Empty

def normalize_exp(x, sigma):
    """ Normalizes the value using an exponential """
    return 1.0 - math.exp(-math.pow(x, 2) / sigma)

class HumanDynamics():
    def __init__(self, init_human_state, TS, params = {'robot_repulsion': 0.5,
                                 'goal_attraction': 0.5,
                                 'drag': 0.5}):
        self.prev_human_state = State(position,velocity, acceleration)
        self.TS = TS
        self.init_human_state = init_human_state
        self.goal_pos = goal_pos
        self.params = params
        self.reset_human_sub = rospy.Subscriber("reset_human",Empty, self._reset_human)
    def get_position(self, bodypoints)
        #Simulate the human as a point mass
        #use forward euler to calculate next state
        for bodypoint in bodypoints:
           F_repulse += self.norm(self.prev_human_state.position-bodypoint)/params['robot_repulsion']^2
        F_attract = self.norm(self.prev_human_state.position - self.goal_pos)/params['goal_attraction']^2
        F_drag = self.prev_human_state.velocity*params["drag"]
        F_total = F_repulse + F_attract + F_drag
        acc = F_total #assume mass is 1, the params need to be tuned anyway
        next_state.velocity = acc*self.TS+self.prev_state.velocity
        next_state.position = self.prev_state.velocity*self.TS + self.prev_state.position

        return next_state
    def get_simple_position(self, bodypoints):
        next_state.position = self.TS*self.prev_human_state.velocity
        next_state.velocity = self.prev_human_state.velocity
        return next_state
    
    def _reset_human(self)
        self.prev_state = self.init_human_state

    def _norm(self, a, b):
        """ takes the two norm of two 3DOF vectors """
        math.sqrt(math.pow(eef_pose[4] - self.human_pose.position.x, 2) +
                         math.pow(eef_pose[5] - self.human_pose.position.y, 2) +
                         math.pow(eef_pose[6] - self.human_pose.position.z, 2))

class test_human_dyanmics():
    __init__(self):
        self.counter = 0
    def get_position(self, reset):
        if reset:
            self.counter = 0
        else 
            self.couter = counter+1
        return self.counter

class WaypointCostFunction(CostFunction):
    def __init__(self, robot, eef_link_name='j2s7s300_end_effector'):
        CostFunction.__init__(self, params={'hit_human_penalty': 0.5,
                                            'normalize_sigma': 1.0})
        #self.human = HumanDynamics()
        self.test_human_dyanmics()
        self.robot = robot
        self.hit_human_penalty = .5
        self.eef_link_name = eef_link_name
        self._care_about_distance = .1
        self.tf_listener = TransformListener()    
        self.human_pose = Pose(Point(0.4,0.4,0.85),Quaternion(0,0,0,1))

        self.human_position_sub = rospy.Subscriber("human_state", Pose, self._get_position)

    def _get_position(self, human_pose):
        pass
        #self.human_pose = human_pose
        #rospy.loginfo("current pose {}".format(human_pose))

    def get_cost(self, config):
        """define the cost for a given configuration"""
        q = config.tolist()
        # q[2] += math.pi  # TODO this seems to be a bug in OpenRAVE?
        self.robot.SetDOFValues(q + [0.0, 0.0, 0.0])
        eef_link = self.robot.GetLink(self.eef_link_name)
        if eef_link is None:
            print("Error: end-effector \"{}\" does not exist".format(self.eef_link_name))
            return 0.0
        bodypoints = []
        eef_pose = openravepy.poseFromMatrix(eef_link.GetTransform())
        bodypoints.append(eef_pose)
        cost = 0.0
        # Get the (normalized) distance from the end-effector to the waypoint
        distance = self._dist(bodypoints)
        if abs(distance) < self._care_about_distance:
            norm_distance = normalize_exp(distance, sigma=self.params['normalize_sigma'])
            # assign cost inverse proportional to the distance squared 
            # TODO swap this with something more principled
            cost += self.params['hit_human_penalty'] * 1/math.pow(distance,2)
        return cost
   
    def get_cost_func(self, t):
        self.human.get_simple_position(t)
        return self.get_cost(config)
    
    def _dist(self, bodypoints):
        human_pos = self.human.get_simple_state(bodypoints).position
        for bodypoint in bodypoints 
            dist += math.sqrt(math.pow(bodypoint[4] - self.human_pos.x, 2) +
                            math.pow(bodypoint[5] - self.human_pos.y, 2) +
                            math.pow(bodypoint[6] - self.human_pos.z, 2))
        return dist
   
    def get_waypoint_markers(self):
        markers = MarkerArray()

        # Choose a random color for this body
        color_r = random.random()
        color_g = random.random()
        color_b = random.random()
        rospy.loginfo("current pose {}".format(self.human_pose))
        waypoint_marker = Marker()
        waypoint_marker.header.frame_id = '/world'
        waypoint_marker.header.stamp = rospy.get_rostime()
        waypoint_marker.ns = '/waypoint'
        waypoint_marker.id = 1
        waypoint_marker.type = Marker.SPHERE
        waypoint_marker.pose = self.human_pose
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
        text_marker.id = 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.pose = self.human_pose
        text_marker.scale.z = 0.05
        text_marker.color.r = color_r
        text_marker.color.g = color_g
        text_marker.color.b = color_b
        text_marker.color.a = 0.50
        text_marker.text = 'planning around human located here'
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

        self.make6DofMarker(False, InteractiveMarkerControl.MOVE_ROTATE_3D, True )
        self.server.applyChanges()

        self.start_human_pub = rospy.Publisher("start_human", Empty, queue_size=10)
        self.reset_human_pub = rospy.Publisher("reset_human", Empty, queue_size=10)

    def _onclick_callback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            if feedback.menu_entry_id == 1:
                self.start_human_pub.publish(Empty())
                #Publish empty message to start human topic
                self._plan_and_execute(feedback)

            elif feedback.menu_entry_id == 2:
                self.reset_human_pub.publish(Empty())
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