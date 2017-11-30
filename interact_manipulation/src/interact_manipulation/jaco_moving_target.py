#!/usr/bin/python
import math
import openravepy
import rospy
import random
import copy
import time
import numpy as np

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
import moveit_commander
from jaco import JacoInterface
from jaco_trajopt import CostFunction
from geometry_msgs.msg import *
from tf import TransformBroadcaster, TransformListener
from math import sin
from std_msgs.msg import Empty


class HumanState():
    """Stores the state of a human
        @Param position: Human position (3,) numpy array
        @Param velocity: Human velocity (3,) numpy array
    """
    def __init__(self, position, velocity):
        """

        """
        self.position = position
        self.velocity = velocity

    @property
    def position(self):
        return self.position
    
    @position.setter
    def position(self, pos):
        self.position = pos
    
    @property
    def velocity(self):
        return self.velocity

    @velocity.setter
    def velocity(self, vel):
        self.velocity = vel
    
    def __str__(self):
        pos = self.position
        vel = self.velocity
        return '({},{})'.format(pos, vel)
    
    __repr__ = __str__

class HumanModel():
    """ Simulates a human for use in path planning (trajopt)
        @Param start_state: starting HumanState object
        @Param goal_pose: The known goal_pose of the human (3,1) numpy array
        @param simulation method: The method in which to simulate the human e.g. constant velocity
        @param dt: the fixed time between waypoints (default 0.2)
    """
    def __init__(self, start_state, goal_pose, simulation_method, dt=.2, params = {'goal_attraction': 0.05, 
                                                                        'robot_repulsion': 0.05,
                                                                        'drag': 0.04,}):
        self.start_state = copy.deepcopy(start_state)
        self.goal_pose = goal_pose
        self.simulation_method = simulation_method
        self.dt = dt
        self.params = params
        
        self.human_positions = []
        self.current_state = copy.deepcopy(start_state) #TODO I need to copy this...I cant just pass it in
        self.human_positions.append(start_state.position)

    def reset_model(self):
        """ Reset the model to prepare for another forward simulation
        """
        self.human_positions = []
        self.current_state = self.start_state
        self.human_positions.append(self.start_state.position)
        rospy.loginfo("resetting states {} start state is now {}".format(self.human_positions,self.start_state))

    def get_human_positions(self, eef_positions):
        """ Return the predicted positions of the human
            @Param eef_positions: a list of (3,) numpy arrays 
        """
        for eef_position in eef_positions:
            advance_model = getattr(self, self.simulation_method)
            advance_model(eef_position)
        return self.human_positions
    
    def constant_velocity(self, eef_position):
        """ Evolve the human state forward using a constant velocity assumption
        """
        self.current_state.position += self.current_state.velocity*self.dt
        print(self.current_state.position)
        self.human_positions.append(self.current_state.position)

class WaypointCostFunction(CostFunction):
    def __init__(self, robot, eef_link_name='j2s7s300_end_effector'):
        CostFunction.__init__(self, params={'hit_human_penalty': 0.5,
                                            'normalize_sigma': 1.0,
                                            'care_about_distance': 0.1})
        self.robot = robot
        human_start_position = np.array([-0.5,0.216,0.538])
        self.human_goal_position = np.array([0,0.216,0.538])
        human_velocity = np.array([.01,0,0])
        self.human_start_state = HumanState(np.array([-0.5,0.216,0.538]), np.array([.01,0,0]))
        self.human_model = HumanModel(self.human_start_state, self.human_goal_position, simulation_method="constant_velocity")

        self.eef_link_name = eef_link_name
        self.tf_listener = TransformListener()
        self.robotDOF = 7

    def get_cost(self, configs):
        """ Returns cost based on the distance between the end effector and the human
            this funciton is given as a callback to trajopt
            @param configs: a list with (number of robot dof x num way points)
            given as a callback from trajopt
            @Return cost: A floating point cost value to be optimized by trajopt
        """
        #reshape the list into a ()
        configs = np.asarray(configs)
        configs = np.reshape(configs, (self.robotDOF,-1))
        
        eef_positions = []
        #use a for loop because I need to calculate kinematics one at a time
        for i in range(np.shape(configs)[1]):
            config = configs[:,i]
            eef_positions.append(self.get_OpenRaveFK(config, self.eef_link_name))
        human_positions = self.human_model.get_human_positions(eef_positions)
        self.human_model.reset_model()

        rospy.loginfo("human_positions {}".format(human_positions))
        
        #initialize distances to nans, if I accidentally don't fill one I'll get an error
        distances = np.empty((len(human_positions),))
        distances[:] = np.nan

        cost = 0.0
        for i, (human_position, eef_position) in enumerate(zip(human_positions, eef_positions)):
            #rospy.loginfo("human_position {}".format(human_position))
            distance = np.linalg.norm(human_position - eef_position)
            if distance < self.params['care_about_distance']:
                # assign cost inverse proportional to the distance squared 
                # TODO swap this with something more principled
                cost += self.params['hit_human_penalty'] * 1/(distance**2)
        return cost

    def get_OpenRaveFK(self, config, link_name):
        """ Calculate the forward kinematics using openRAVE for use in cost evaluation.
            @Param config: Robot joint configuration (3,) numpy array
            @Param link_name: Name of the link to calculate forward kinematics for
        """
        q = config.tolist()
        self.robot.SetDOFValues(q + [0.0, 0.0, 0.0])
        eef_link = self.robot.GetLink(link_name)
        if eef_link is None:
            rospy.logerror("Error: end-effector \"{}\" does not exist".format(self.eef_link_name))
            raise ValueError("Error: end-effector \"{}\" does not exist".format(self.eef_link_name))
        eef_pose = openravepy.poseFromMatrix(eef_link.GetTransform())
        return np.array([eef_pose[4], eef_pose[5], eef_pose[6]])

    def get_waypoint_markers(self):
        pass
        # markers = MarkerArray()

        # # Choose a random color for this body
        # color_r = random.random()
        # color_g = random.random()
        # color_b = random.random()
        # rospy.loginfo("current pose {}".format(self.human_start_pose))
        # waypoint_marker = Marker()
        # waypoint_marker.header.frame_id = '/world'
        # waypoint_marker.header.stamp = rospy.get_rostime()
        # waypoint_marker.ns = '/waypoint'
        # waypoint_marker.id = 1
        # waypoint_marker.type = Marker.SPHERE
        # waypoint_marker.pose = self.human_start_state.position
        # waypoint_marker.scale.x = 0.05
        # waypoint_marker.scale.y = 0.05
        # waypoint_marker.scale.z = 0.05
        # waypoint_marker.color.r = color_r
        # waypoint_marker.color.g = color_g
        # waypoint_marker.color.b = color_b
        # waypoint_marker.color.a = 0.50
        # waypoint_marker.lifetime = rospy.Duration(0)
        # markers.markers.append(waypoint_marker)

        # text_marker = Marker()
        # text_marker.header.frame_id = '/world'
        # text_marker.header.stamp = rospy.get_rostime()
        # text_marker.ns = '/waypoint/text'
        # text_marker.id = 1
        # text_marker.type = Marker.TEXT_VIEW_FACING
        # text_marker.pose = self.human_start_pose
        # text_marker.scale.z = 0.05
        # text_marker.color.r = color_r
        # text_marker.color.g = color_g
        # text_marker.color.b = color_b
        # text_marker.color.a = 0.50
        # text_marker.text = 'Human starts here'
        # text_marker.lifetime = rospy.Duration(0)
        # markers.markers.append(text_marker)

        # # end pose
        # color_r = random.random()
        # color_g = random.random()
        # color_b = random.random()
        # end_marker = Marker()
        # end_marker.header.frame_id = '/world'
        # end_marker.header.stamp = rospy.get_rostime()
        # end_marker.ns = '/waypoint/end_location'
        # end_marker.id = 2
        # end_marker.type = Marker.SPHERE
        # end_marker.pose = self.human_goal_pose
        # end_marker.scale.x = 0.05
        # end_marker.scale.y = 0.05
        # end_marker.scale.z = 0.05
        # end_marker.color.r = color_r
        # end_marker.color.g = color_g
        # end_marker.color.b = color_b
        # end_marker.color.a = 0.50
        # end_marker.lifetime = rospy.Duration(0)
        # markers.markers.append(end_marker)

        # return markers
    
class InteractiveMarkerAgent():
    def __init__(self, server_name, position, menu_labels=[], base_frame = "root", scale = .15):
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
        initial_position = Point(-0.5,0.216,0.538)
        menu_label_list = []
        menu_label_list.append("Plan and Execute")
        menu_label_list.append("Reset Human")
        menu_label_list.append("Reset Robot")
        InteractiveMarkerAgent.__init__(self, "End_Goal", initial_position, menu_labels=menu_label_list)

        self.jaco_interface = JacoInterface()
        self.waypoint_cost_func = WaypointCostFunction(self.jaco_interface.planner.jaco)

        self.make6DofMarker(False, InteractiveMarkerControl.MOVE_ROTATE_3D, True )
        self.server.applyChanges()

        self.start_human_pub = rospy.Publisher("start_human", Pose, queue_size=10)
        self.reset_human_pub = rospy.Publisher("reset_human", Pose, queue_size=10)
        self.human_start_pose = Pose(Point(-0.5,0.216,0.538),Quaternion(0,0,0,1))
    def _onclick_callback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            if feedback.menu_entry_id == 1:
                #Publish empty message to start human topic
                self._plan_and_execute(feedback)

            elif feedback.menu_entry_id == 2:
                self.reset_human_pub.publish(self.human_start_pose)
            elif feedback.menu_entry_id == 3:
                self.jaco_interface.home()

        self.server.applyChanges()

    def _plan_and_execute(self, feedback):
        self.jaco_interface.planner.cost_functions = [self.waypoint_cost_func]
        self.jaco_interface.planner.trajopt_num_waypoints = 20

        goal_pose = PoseStamped()
        goal_pose.header = feedback.header
        goal_pose.pose = feedback.pose

        start_pose = self.jaco_interface.arm_group.get_current_pose()
        rospy.loginfo("Requesting plan from start_pose:\n {} \n goal_pose:\n {}".format(start_pose, goal_pose))

        traj = self.jaco_interface.plan(start_pose, goal_pose)
        #rospy.loginfo("Executing trajectory ******* {}".format(traj))
        #m = self.waypoint_cost_func.get_waypoint_markers()
        #self.jaco_interface.marker_array_pub.publish(m)

        #self.start_human_pub.publish(self.human_start_pose)
        self.jaco_interface.execute(traj)

    def run(self):        
        self.jaco_interface.home()

        rospy.spin()

    def make6DofMarker(self, fixed, interaction_mode, show_6dof = False):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self._base_frame
        int_marker.pose.position = self.marker_position
        int_marker.pose.orientation = Quaternion(.707,0,0,-.707)
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