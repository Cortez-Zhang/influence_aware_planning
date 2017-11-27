#!/usr/bin/python
import math
import openravepy
import rospy
import random
import copy
import time

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
    def __init__(self, point, velocity):
        self.position = point
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

class HumanModel():
    def __init__(self, start_state, goal_pose, velocity, dt=.2, params = {'goal_attraction': 0.05, 
                                                                        'robot_repulsion': 0.05,
                                                                        'drag': 0.04,
                                                                        'min_call_time': 16.69}):
        self.start_state = start_state
        self.goal_pose = goal_pose

        self.dt = dt
        self.prev_state = start_state
        self.next_state = start_state
        self.states = []
        self.states.append(start_state)
        self.prev_call_time = time.time()
        self.params = params

    def reset_model(self):
        self.prev_state = self.start_state
        self.states = []
    
    def get_pose(self, t):
        try:
            state = self.states[t]
            rospy.loginfo("Getting/returning human pose x:{} y:{} z:{}".format(state.position.x, state.position.y, state.position.z))
        except:
            state = []
            rospy.logerr("Bad get_pose request for t: {} on states: {}".format(t, self.states))
        return state

    def advance_model(self, bodypoints, t):     
        #Simulate the human as a point mass
        #use forward euler to calculate next state
        rospy.loginfo("advancing time: {}".format(t))
        rospy.loginfo("human located at: x: {} y: {} z: {}".format(self.prev_state.position.x, self.prev_state.position.y, self.prev_state.position.z))
        current_time = time.time()
        delta_time = current_time-self.prev_call_time
        rospy.loginfo("delta_time {}".format(delta_time))
        if delta_time > self.params['min_call_time']:
            F_repulse = Point(0, 0, 0)
            for bodypoint in bodypoints:
                rospy.loginfo("bodypoint located at x: {} y: {} z: {}".format(bodypoint[4], bodypoint[5], bodypoint[6]))

                xdistb = bodypoint[4]-self.prev_state.position.x
                ydistb = bodypoint[5]-self.prev_state.position.y
                zdistb = bodypoint[6]-self.prev_state.position.z

                if abs(xdistb) < 0.05:
                    xdistb = 200
                if abs(ydistb) < 0.05:
                    ydistb = 200
                if abs(zdistb) < 0.05:
                    zdistb = 200
                
                F_repulse.x += -1*self.params['robot_repulsion']/(xdistb)**2
                F_repulse.y += -1*self.params['robot_repulsion']/(ydistb)**2
                F_repulse.z += -1*self.params['robot_repulsion']/(zdistb)**2
                rospy.loginfo("F_repulse x: {} y: {} z: {}".format(F_repulse.x, F_repulse.y, F_repulse.z))

            
            F_attract = Point(0,0,0)

            xdistg = self.goal_pose.x-self.prev_state.position.x
            ydistg = self.goal_pose.y-self.prev_state.position.y
            zdistg = self.goal_pose.z-self.prev_state.position.z

            if abs(xdistg) < 0.05:
                xdistg = 200
            if abs(ydistg) < 0.05:
                ydistg = 200
            if abs(zdistg) < 0.05:
                zdistg = 200

            F_attract.x = self.params['goal_attraction']/(xdistg)**2
            F_attract.y = self.params['goal_attraction']/(ydistg)**2
            F_attract.z = self.params['goal_attraction']/(zdistg)**2
            
            rospy.loginfo("F_attract x: {} y: {} z: {}".format(F_attract.x, F_attract.y, F_attract.z))

            F_drag = Point(0,0,0)
            F_drag.x = -self.prev_state.velocity.x**2*self.params["drag"]
            F_drag.y = -self.prev_state.velocity.y**2*self.params["drag"]
            F_drag.z = -self.prev_state.velocity.z**2*self.params["drag"]
            rospy.loginfo("F_drag x: {} y: {} z: {}".format(F_drag.x, F_drag.y, F_drag.z))

            acc = Point(0,0,0)  #assume mass is 1, the params need to be tuned anyway
            acc.x = F_repulse.x + F_attract.x + F_drag.x
            acc.y = F_repulse.y + F_attract.y + F_drag.y 
            acc.z = F_repulse.z + F_attract.z + F_drag.z
            
            self.next_state.velocity.x = acc.x*self.dt+self.prev_state.velocity.x
            self.next_state.velocity.y = acc.y*self.dt+self.prev_state.velocity.y
            self.next_state.velocity.z = acc.z*self.dt+self.prev_state.velocity.z
            rospy.loginfo("prev_state.vel x: {} y: {} z: {}".format(self.prev_state.velocity.x, self.prev_state.velocity.y, self.prev_state.velocity.z))

            
            self.next_state.position.x = self.prev_state.velocity.x*self.dt + self.prev_state.position.x
            self.next_state.position.y = self.prev_state.velocity.y*self.dt + self.prev_state.position.y
            self.next_state.position.z = self.prev_state.velocity.z*self.dt + self.prev_state.position.z
            
            rospy.loginfo("Advancing human at t: {} to state: {}".format(t,self.next_state))
            self.prev_state = self.next_state
            self.states.append(self.next_state)

        else:
            rospy.loginfo("Did not advance human at t: {}")
    
    def _norm_sqrd(self, vel):
        return math.sqrt(vel.x**2, vel.y**2, vel.z**2)

    #TODO this function is redundnat with the _dist func below
    def _dist1(self, bodypoint, human_pos):

        return math.sqrt(math.pow(bodypoint[4] - human_pos.x, 2) +
                         math.pow(bodypoint[5] - human_pos.y, 2) +
                         math.pow(bodypoint[6] - human_pos.z, 2))
    
    def _dist2(self, goal_pos, human_pos):

        return math.sqrt(math.pow(goal_pos.x - human_pos.x, 2) +
                         math.pow(goal_pos.y - human_pos.y, 2) +
                         math.pow(goal_pos.z - human_pos.z, 2))
        
    def get_simple_pose(self, t):
        #print("t in get_pose: {} start_pose {}".format(t, self.start_pose))
        x = self.start_pose.position.x+self.velocity.linear.x*t
        y = self.start_pose.position.y+self.velocity.linear.y*t
        z = self.start_pose.position.z+self.velocity.linear.z*t
        pose = Pose(Point(x,y,z), Quaternion(0,0,0,1))
        #print("returning pose in get pose: {}".format(pose))
        return pose
    
    def get_pose_forPlayBack(self,t):

        return self.state[t]

class CostWithTime():
    def __init__(self, t, get_cost_func):
        self.time = t
        self.get_cost_func = get_cost_func
    def __call__(self, config):
        #print("calling cost with time {}".format(self.time))        
        return self.get_cost_func(config, self.time)

class WaypointCostFunction(CostFunction):
    def __init__(self, robot, eef_link_name='j2s7s300_end_effector'):
        CostFunction.__init__(self, params={'hit_human_penalty': 0.5,
                                            'normalize_sigma': 1.0,
                                            'care_about_distance': 0.1})
        self.robot = robot
        self.human_start_pose = Pose(Point(-0.5,0.216,0.538),Quaternion(0,0,0,1))
        self.human_goal_pose = Pose(Point(0,0.216,0.538), Quaternion(0,0,0,1))
        self.human_velocity = Vector3(0,0,0)
        self.human_start_state = HumanState(self.human_start_pose.position, self.human_velocity)
        self.human_model = HumanModel(self.human_start_state, self.human_goal_pose.position, self.human_velocity)

        self.eef_link_name = eef_link_name
        self.tf_listener = TransformListener()    

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
        bodypoints = [eef_pose] #TODO add more links here
        
        if t == 0:
            self.human_model.reset_model()
        else:
            self.human_model.advance_model(bodypoints, t)
        
        human_pose = self.human_model.get_pose(t)
        distance = self._dist(eef_pose, human_pose)
        if distance < self.params['care_about_distance']:
            # assign cost inverse proportional to the distance squared 
            # TODO swap this with something more principled
            cost += self.params['hit_human_penalty'] * 1/math.pow(distance,2)

        return cost

    def _dist(self, eef_pose, human_pose):
        human_pos = human_pose.position
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

        # end pose
        color_r = random.random()
        color_g = random.random()
        color_b = random.random()
        end_marker = Marker()
        end_marker.header.frame_id = '/world'
        end_marker.header.stamp = rospy.get_rostime()
        end_marker.ns = '/waypoint/end_location'
        end_marker.id = 2
        end_marker.type = Marker.SPHERE
        end_marker.pose = self.human_goal_pose
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
        #rospy.loginfo("Executing trajectory ******* {}".format(traj))
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