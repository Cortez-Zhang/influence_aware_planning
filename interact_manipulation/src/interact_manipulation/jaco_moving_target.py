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
from geometry_msgs.msg import *
from tf import TransformBroadcaster, TransformListener
from math import sin
from std_msgs.msg import Empty

def normalize_exp(x, sigma):
    """ Normalizes the value using an exponential """
    return 1.0 - math.exp(-math.pow(x, 2) / sigma)

class HumanModel():
    def __init__(self, start_pose, velocity, dt=.2):
        self.start_pose = start_pose
        #print("start pose initialized: {}".format(self.start_pose))
        self.velocity = velocity
        self.dt = dt
        #print("intial velcoity initialized {}".format(velocity))


    def get_pose(self, config,t):
        #print("t in get_pose: {} start_pose {}".format(t, self.start_pose))
        x = self.start_pose.position.x+self.velocity.linear.x*t
        y = self.start_pose.position.y+self.velocity.linear.y*t
        #print("self.velocity.linear.y {}".format(self.velocity.linear.y))
        #print("next_y: {} self.start_pose.position.y: {}".format(y, self.start_pose.position.y))
        z = self.start_pose.position.z+self.velocity.linear.z*t
        pose = Pose(Point(x,y,z), Quaternion(0,0,0,1))
        #print("returning pose in get pose: {}".format(pose))
        return pose
    
    def get_pose_forPlayBack(self,time_from_start):
        remaining_time = time_from_start%self.dt
        t1 = math.floor(time_from_start/self.dt)
        #rospy.loginfo("t1: {}".format(t1))

        if remaining_time !=0:
            percent_between = remaining_time/self.dt
            pose1 = self.get_pose([], t1)
            t2 = math.ceil(time_from_start/self.dt)
            pose2 = self.get_pose([], t2)
            #rospy.loginfo("t2: {}".format(t2))
            #rospy.loginfo("pose1 {}".format(pose1))
            #rospy.loginfo("pose2 {}".format(pose2))
            #rospy.loginfo("percent_between: {}".format(percent_between))
            pose = self._interpolate(percent_between,pose1,pose2)
            #rospy.loginfo("interpolated pose: {}".format(pose))
        else:
            pose = self.get_pose([], t1)
            #rospy.loginfo("no remainder, returning pose for t1: {}".format(pose))
        return pose

    def _interpolate(self, percent_between, pose1, pose2):
        pose = Pose()
        pose.orientation = Quaternion(0,0,0,1)
        pose.position.x = pose1.position.x*percent_between + pose2.position.x*(1-percent_between)
        pose.position.y = pose1.position.y*percent_between + pose2.position.y*(1-percent_between)
        pose.position.z = pose1.position.z*percent_between + pose2.position.z*(1-percent_between)
        return pose

    #TODO this is obsolete, delete?
    def get_end_waypoint_marker(self):
        markers = MarkerArray()

        return markers

class CostWithTime():
    def __init__(self, t, get_cost_func):
        self.time = t
        #print("setting up cost with time t: {}".format(t))
        self.get_cost_func = get_cost_func
        #print("init time {}".format(self.time))
    def __call__(self, config):
        #print("calling cost with time {}".format(self.time))        
        return self.get_cost_func(config, self.time)

class WaypointCostFunction(CostFunction):
    def __init__(self, robot, eef_link_name='j2s7s300_end_effector'):
        CostFunction.__init__(self, params={'hit_human_penalty': 0.5,
                                            'normalize_sigma': 1.0})
        self.robot = robot
        self.human_start_pose = Pose(Point(-0.5,0.216,0.538),Quaternion(0,0,0,1))
        human_velocity = Twist(Vector3(.05,0,0),Vector3(0,0,0))
        self.human_model = HumanModel(self.human_start_pose, human_velocity, .2)
        end_time = 10
        self.human_end_pose = self.human_model.get_pose(0,end_time)
        self.hit_human_penalty = .5
        self.eef_link_name = eef_link_name
        self._care_about_distance = .1

        self.tf_listener = TransformListener()    

        self.human_position_sub = rospy.Subscriber("human_state", Pose, self._get_position)

    def _get_position(self, human_pose):
        pass
        #self.human_pose = human_pose
        #rospy.loginfo("current pose {}".format(human_pose))
    def get_cost_func(self, t):
        #print("setting up cost func t= {}".format(t))
        return CostWithTime(t, self.get_cost_with_t)

    def get_cost(self, config):
        pass

    def get_cost_with_t(self, config, t):
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
        rospy.loginfo("time: {}".format(t))
        rospy.loginfo("config: {}".format(q))
        #print("time in get_cost_with_t {}".format(t))
        #print("q: {}".format(q))
        # Get the (normalized) distance from the end-effector to the waypoint
        human_pose = self.human_model.get_pose(config, t)
        distance = self._dist(eef_pose, human_pose)
        if abs(distance) < self._care_about_distance:
            # assign cost inverse proportional to the distance squared 
            # TODO swap this with something more principled
            cost += self.params['hit_human_penalty'] * 1/math.pow(distance,2)

        return cost

    def _dist(self, eef_pose, human_pose):
        #(trans,rot) = self.tf_listener.lookupTransform('/root', '/moving_frame', rospy.Time.now())
        human_pos = human_pose.position
        #print("human_pos.y: {}".format(human_pos.y))
        #print("eef_pose[5] {}".format(eef_pose[5]))
        return math.sqrt(math.pow(eef_pose[4] - human_pos.x, 2) +
                         math.pow(eef_pose[5] - human_pos.y, 2) +
                         math.pow(eef_pose[6] - human_pos.z, 2))

    def get_waypoint_markers(self):
        markers = MarkerArray()

        # Choose a random color for this body
        color_r = random.random()
        color_g = random.random()
        color_b = random.random()
        rospy.loginfo("current pose {}".format(self.human_start_pose))
        waypoint_marker = Marker()
        waypoint_marker.header.frame_id = '/world'
        waypoint_marker.header.stamp = rospy.get_rostime()
        waypoint_marker.ns = '/waypoint'
        waypoint_marker.id = 1
        waypoint_marker.type = Marker.SPHERE
        waypoint_marker.pose = self.human_start_pose
        waypoint_marker.scale.x = 0.05
        waypoint_marker.scale.y = 0.05
        waypoint_marker.scale.z = 0.05
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
        text_marker.pose = self.human_start_pose
        text_marker.scale.z = 0.05
        text_marker.color.r = color_r
        text_marker.color.g = color_g
        text_marker.color.b = color_b
        text_marker.color.a = 0.50
        text_marker.text = 'Human starts here'
        text_marker.lifetime = rospy.Duration(0)
        markers.markers.append(text_marker)

        arrow_marker = Marker()
        arrow_marker.header.frame_id = '/world'
        arrow_marker.header.stamp = rospy.get_rostime()
        arrow_marker.ns = '/waypoint/arrow'
        arrow_marker.id = 1
        arrow_marker.type = Marker.ARROW
        
        arrow_marker.pose = Pose(self.human_start_pose.position,Quaternion(0,0,0,1))
        arrow_marker.scale.x = .1
        arrow_marker.scale.y = 0.05
        arrow_marker.scale.z = 0.05
        arrow_marker.color.r = 1
        arrow_marker.color.b = 0
        arrow_marker.color.g = 0
        arrow_marker.color.a = 0.5
        arrow_marker.lifetime = rospy.Duration(0)
        markers.markers.append(arrow_marker)

        # end pose
        color_r = random.random()
        color_g = random.random()
        color_b = random.random()
        rospy.loginfo("current pose {}".format(self.human_start_pose))
        end_marker = Marker()
        end_marker.header.frame_id = '/world'
        end_marker.header.stamp = rospy.get_rostime()
        end_marker.ns = '/waypoint/end_location'
        end_marker.id = 2
        end_marker.type = Marker.SPHERE
        end_marker.pose = self.human_end_pose
        end_marker.scale.x = 0.05
        end_marker.scale.y = 0.05
        end_marker.scale.z = 0.05
        end_marker.color.r = color_r
        end_marker.color.g = color_g
        end_marker.color.b = color_b
        end_marker.color.a = 0.50
        end_marker.lifetime = rospy.Duration(0)
        markers.markers.append(end_marker)

        return markers
    
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
        rospy.loginfo("Executing trajectory ******* {}".format(traj))
        m = self.waypoint_cost_func.get_waypoint_markers()
        self.jaco_interface.marker_array_pub.publish(m)

        self.start_human_pub.publish(self.human_start_pose)
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