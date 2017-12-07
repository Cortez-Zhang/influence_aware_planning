#!/usr/bin/python
import math
import openravepy
import rospy
import random
import copy
import time
import numpy as np
import datetime

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
import moveit_commander
from jaco import JacoInterface
from jaco_trajopt import CostFunction
from geometry_msgs.msg import *
from tf import TransformBroadcaster, TransformListener
from math import sin
from std_msgs.msg import Empty, Float32MultiArray, MultiArrayLayout, MultiArrayDimension

import marker_wrapper

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
        Params
        ---
        start_state: starting HumanState object
        goals: A list of (3,) numpy arrays, known goal_positions of the human
        simulation method: The method in which to simulate the human 
        e.g. constant velocity, point_mass, certainty_based_speed
        dt: the fixed time between waypoints (default 0.2)
    """
    def __init__(self, start_state, goals, simulation_method, goal_inference=None, dt=.2, params = {'mass': .006, 
                                                                        'robot_aggressiveness': .3,
                                                                        'drag': 0,
                                                                        'force_cap': 0.02,
                                                                        'max_certainty_speed': 0.2}):
        self.start_state = copy.deepcopy(start_state)
        self.goals = goals
        self.simulation_method = simulation_method
        self.dt = dt
        self.params = params
        self.goal_inference = goal_inference #GoalInference instance, set to none unless simulation_method == certainty_based_speed
        
        self.human_positions = []
        self.current_state = copy.deepcopy(start_state)
        self.human_positions.append(start_state.position)

        marker_wrapper.show_position_marker(label="human \n start\n\n", position = start_state.position, ident=1, color=(1,0,0))
        for i, goal in enumerate(goals):
           marker_wrapper.show_position_marker(label="human \n goal\n\n", position= goal, ident=10+i, color=(0,1,0))
    
    def reset_model(self):
        """ Reset the model to prepare for another forward simulation
        """
        if self.goal_inference:
            self.goal_inference.reset()
        self.human_positions = []
        self.human_velocities = []
        self.current_state = copy.deepcopy(self.start_state)
        self.human_positions.append(self.start_state.position)

    def get_human_positions(self, eef_positions):
        """ Get the predicted positions of the human for a given set of robot positions
            i.e. how will the human react to the robot?
            Param
            ---
            eef_positions: a list of (3,) numpy arrays containing the positions of the end effector
        """
        advance_model = getattr(self, self.simulation_method)
        
        prev_eef_pos = eef_positions[0]
        for eef_pos in eef_positions:
            advance_model(eef_pos, prev_eef_pos)
            prev_eef_pos = eef_pos.copy()

        return self.human_positions    
            
    def constant_velocity(self, eef_position, prev_eef_pos):
        """ Evolve the human state forward using a constant velocity assumption
        """
        #rospy.loginfo("current_state {}".format(self.current_state))
        curr_pos = self.current_state.position
        next_pos = curr_pos + self.current_state.velocity*self.dt
        self.current_state.position = next_pos

        self.human_positions.append(next_pos)
        
    def point_mass(self, eef_position, prev_eef_pos):
        """ Evolve the human state forward using a point mass model
            We assume the human will move away from the robot and towards its goal
        """
        curr_pos = self.current_state.position
        curr_vel = self.current_state.velocity

        F_repulse = -1*self.params["robot_aggressiveness"]*self.potential_field(eef_position,curr_pos)
        
        F_attract = 0.0
        for goal in goals:
            F_attract+= (1-self.params["robot_aggressiveness"])*self.potential_field(self.goal,curr_pos)
        
        acc = (F_attract+F_repulse)*self.params["mass"]

        next_vel = curr_vel+acc*self.dt
        next_pos = 0.5*acc*self.dt**2+curr_vel*self.dt+curr_pos

        self.current_state.position = next_pos
        self.current_state.velocity = next_vel

        self.human_velocities.append(next_vel)
        self.human_positions.append(next_pos)
    
    def speed_based_certainty(self, eef_position, prev_eef_pos):
        """ Human moves at a speed proportional to belief over robot goals
            The human will move to the goal the robot is not going towards
            the human moves faster if it is more certain that is the goal
            Param
            ---
            eef_position: a (3,) numpy array with xyz position of robot end effector
        """
        #TODO what happens if I am at the goal?
        curr_pos = self.current_state.position
        dist_goal1 = np.linalg.norm(curr_pos - self.goals[0])
        dist_goal2 = np.linalg.norm(curr_pos - self.goals[1])
        #rospy.loginfo("dist_goal1 {}, dist_goal2 {}".format(dist_goal1, dist_goal2))
        if dist_goal1>0.05 and dist_goal2>0.05:
            self.goal_inference.update(eef_position,prev_eef_pos)
            b = self.goal_inference.current_beliefs

            #TODO there can only be two goals expand so it can be more
            speed = self.params["max_certainty_speed"]*(min(b)*2)
            human_goal = b.index(min(b))
            goal_dir = GoalInference.direction(curr_pos,self.goals[human_goal]) #humans goal direction
            
            next_vel = speed*goal_dir
            next_pos = next_vel*self.dt+curr_pos
            
            self.current_state.position = next_pos
            self.current_state.velocity = next_vel
            
        else:
            next_pos = self.current_state.position
            next_vel = self.current_state.velocity
            #self.human_velocities
        self.human_velocities.append(next_vel.copy())
        self.human_positions.append(next_pos.copy())

    def potential_field(self, obstacle, curr_pos):
        """ Calculate distance penalty for obstacles
            Params
            ---
            obstacle: a (3,) numpy array with position of obstacle or goal
            curr_pos: a (3,) numpy array with position of human
        """ 
        #TODO vectorize this using vectorized comparisons
        epsilon = self.params["force_cap"]

        force = np.empty((3,))

        dist = obstacle-curr_pos
        dist_norm = np.linalg.norm(dist)
        direction = dist/dist_norm
        if dist_norm < epsilon:
            dist_norm = epsilon
        
        return direction/dist_norm

class GoalInference(object):
    """ Creates a model which can store and update belief over goals
        Params
        ---
        goals: A list of (3,) numpy arrays
        variance: A float representing the variance of the gaussian default 0.01
        current_beliefs: A list of float priors on goals, default [0.5, 0.5]
        beliefs_over_time: A list of belief lists one for each time
    """
    def __init__(self, goals, variance = 0.01, current_beliefs = [0.5,0.5]):
        self.beliefs_over_time = [] #a list of lists, each belief in time
        self.goals = goals
        self.variance = variance
        self.current_beliefs = current_beliefs #a list of scaler beliefs for each goal

    def reset(self):
        """ Reset the goal inference back to nothing
        """
        self.current_beliefs = [.5,.5]
        self.beliefs_over_time = []

    def gaussian(self, mu):
        """ Compute an N dim independent multivariate gaussian (Covariance terms are 0)
            Params
            ----
            mu: a vector of means
            Returns
            ----
            gaussian: A function which computes the probability density of a 3D point
        """
        cov = self.variance * np.eye(mu.shape[0])
        return lambda x: (1./np.sqrt(2*math.pi*np.linalg.det(cov))) * np.exp(
                -(1./2.) * np.dot(np.dot((x - mu), np.linalg.inv(cov)), (x - mu))
                )
    @staticmethod
    def direction(x, y):
        """ Compute the direction from x to y
            Params
            ---
            x: a numpy array
            y: a numpy array
            Returns
            ---
            normalized direction: numpy array of unit length to y from x
        """
        return (y - x)/np.linalg.norm(y - x + 1e-12)
    
    @staticmethod
    def normalize(beliefs):
        """ Normalize a descrete set of beliefs
            Params
            ---
            beliefs: a list of scaler beliefs #TODO check this
            Returns
            ---
            norm_beliefs: a np array of normalized beliefs
        """
        return np.asarray(beliefs)/np.sum(np.asarray(beliefs))

    def update(self, eef_pos, prev_eef_pos):
        """ Updates the belief over goals
            Params
            ---
            prev_eef_pos: a (3,) numpy array with xyz of robot end effector
            eef_pos: current location of end effector
            Returns
            ---
            norm_beliefs: a list of new beliefs given the observation (eef_pos)
        """
        #print("prev_eef_pos {} curr_eef_pos {}".format(prev_eef_pos,eef_pos))
        goal_dirs = [GoalInference.direction(eef_pos,goal) for goal in self.goals]
        #print("direction to goals: {}".format(goal_dirs))
        #rospy.loginfo("goal_dirs {}".format(goal_dirs))
        interaction_dir = GoalInference.direction(prev_eef_pos,eef_pos)
        #print("interaction_dir: {}".format(interaction_dir))
        #rospy.loginfo("prev_eef_pos {}, eef_pos {}".format(prev_eef_pos, eef_pos))
        #rospy.loginfo("interaction_dir {}")
        beliefs = np.array([b*self.gaussian(goal_dir)(interaction_dir) for (b, goal_dir) in zip(self.current_beliefs, goal_dirs)])
        #print("beliefs {}".format(beliefs))
        norm_belief = GoalInference.normalize(beliefs)
        #print("normalized beliefs {}".format(norm_belief))
        self.beliefs_over_time.append(norm_belief)
        self.current_beliefs = norm_belief.tolist()
#        return norm_belief


class WaypointCostFunction(CostFunction):
    def __init__(self, robot, human_model,eef_link_name='j2s7s300_end_effector'):
        CostFunction.__init__(self, params={'hit_human_penalty': 0.5,
                                            'normalize_sigma': 1.0,
                                            'care_about_distance': 0.1})
        self.robot = robot
        self.human_model = human_model
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
        #rospy.loginfo("configurations {}".format(configs))
        
        eef_positions = []
        #use a for loop because I need to calculate kinematics one at a time
        for i in range(np.shape(configs)[1]):
            config = configs[:,i]
            eef_positions.append(self.get_OpenRaveFK(config, self.eef_link_name))
        
        self.human_model.reset_model()
        human_positions = self.human_model.get_human_positions(eef_positions)
        human_velocities = self.human_model.human_velocities

        #self.human_closeness_cost(human_positions, eef_positions) #TODO add param to specify cost
        #self.human_speed_cost(human_velocities) #TODO add param to specify cost
        #self.human_go_first_cost(human_positions)
        return self.human_speed_cost(human_velocities)
    
    def human_go_first_cost(self, human_positions):
        """Calculate the distance above midpoint for human going first
            This cost can only be used with a single goal. Its meant to be something like an intersection
            The human and robot cross paths and the robot wants the human to go first.
            Params
            ---
            human_positions: list of (3,) numpy arrays
        """
        assert len(self.human_model.goals)==1 
        
        cost = 0.0
        #get midpoint between start and goal
        start_pos = self.human_model.start_state.position
        goal_pos = self.human_model.goals[0]

        midpoint = (start_pos+goal_pos)/2.0
        to_mid = midpoint-start_pos
        norm_to_mid = np.linalg.norm(to_mid)

        for position in human_positions:
            #TODO vectorize this better 
            """" Calculate individual waypoint cost """
            distance=np.dot(to_mid,midpoint-position)/norm_to_mid
            cost = np.tanh(distance)    
        return cost
    
    def human_speed_cost(self, human_velocities):
        cost = 0.0
        for vel in human_velocities:
            speed = np.linalg.norm(vel)
            #rospy.loginfo(speed)
            cost+=speed
        return cost
           
    def human_closeness_cost(self,human_positions,eef_positions):
        cost = 0.0
        for i, (human_position, eef_position) in enumerate(zip(human_positions, eef_positions)):
            #rospy.loginfo("human_position {}".format(human_position))
            distance = np.linalg.norm(human_position - eef_position)
            if distance < self.params['care_about_distance']:
                # assign cost inverse proportional to the distance squared 
                # TODO swap this with something more principled
                cost += self.params['hit_human_penalty'] * 1/(distance)
        #SimplePointSimulator(eef_positions, human_positions, repeat=False).simulate()
        return cost/2.0
        #return cost 

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
    
class InteractiveMarkerAgent():
    def __init__(self, server_name, position, menu_labels=[], base_frame = "root", scale = .15):
        self._scale = scale
        self.server = InteractiveMarkerServer(server_name)
        self.menu_handler = MenuHandler()
        self.menu_labels = menu_labels
        self.setup_menu_handler()
        self.marker_position = Point(position.x, position.y, position.z)
        self.base_frame = base_frame

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

#class data

class AssertiveRobotPlanner(InteractiveMarkerAgent):
    def __init__(self):
        initial_position = Point(-0.5,0.216,0.538) #initial position of the marker
        menu_label_list = []
        menu_label_list.append("Plan and Execute")
        menu_label_list.append("Reset Human")
        menu_label_list.append("Reset Robot")
        InteractiveMarkerAgent.__init__(self, "End_Goal", initial_position, menu_labels=menu_label_list)

        self.jaco_interface = JacoInterface()
        self.make6DofMarker(False, InteractiveMarkerControl.MOVE_ROTATE_3D, True )
        self.server.applyChanges()

        self.start_human_pub = rospy.Publisher("start_human",Float32MultiArray , queue_size=10)
        self.reset_human_pub = rospy.Publisher("reset_human",Empty, queue_size=10)
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
        """ Callback to interactive marker
            This is where the main stuff happens. 
        """
        self.jaco_interface.home()

        #get the location of the robot and use that as the start pose
        start_pose = self.jaco_interface.arm_group.get_current_pose()

        #create the goal_pose request for the robot
        goal_pose = PoseStamped()
        goal_pose.header = feedback.header
        goal_pose.pose.position = Point(-0.2,-0.2,0.538)
        goal_pose.pose.orientation = start_pose.pose.orientation
        
        goal1 = np.array([-0.4,-0.1,0.538])
        goal2 = np.array([-0.2,-0.2,0.538])
        goals = [goal1, goal2]
        goal_inference = GoalInference(goals, variance = 2)

        #create the human model
        human_start_state = HumanState(np.array([-.3,0.3,0.538]), np.array([0,0,0]))
        human_model = HumanModel(human_start_state, goals, simulation_method="speed_based_certainty", goal_inference=goal_inference)
        
        #create the robot cost function, including human model
        cost_func = WaypointCostFunction(self.jaco_interface.planner.jaco, human_model)
        self.jaco_interface.planner.cost_functions = [cost_func]
        self.jaco_interface.planner.trajopt_num_waypoints = 20
        
        rospy.loginfo("Requesting plan from start_pose:\n {} \n goal_pose:\n {}".format(start_pose, goal_pose))

        #call trajopt
        traj = self.jaco_interface.plan(start_pose, goal_pose)

        marker_wrapper.show_position_marker(human_model.human_positions[-1], label="human end", ident=3)
        
        #package up the human trajectory into a message
        #trajmsg = self._to_trajectory_message(human_model.human_positions)
        
        #send the humna trajectory to an interactive marker for visualization
        #self.start_human_pub.publish(trajmsg)
        #time_before = rospy.get_time()

        robot_positions = traj.joint_trajectory.points
        human_positions = human_model.human_positions
        simulator = SimplePointSimulator(robot_positions, human_positions, self.jaco_interface)
        simulator.simulate()

        #self.jaco_interface.execute(traj)
        #rospy.loginfo("time to execute {}".format(rospy.get_time()-time_before))

    def _to_trajectory_message(self, positions):
        """Convert the human positions to a message that behaves like a trajectory
            to be passed to human.py
            #TODO this function is now obsolete
        """
        float_array = Float32MultiArray()
        
        layout = MultiArrayLayout()
        layout.data_offset = 0

        dims = []
        dim = MultiArrayDimension()
        dim.label = "traj"
        dim.size = 0
        dim.stride = 0
        dims.append(dim)

        layout.dim = dims

        float_list = []
        for position in positions:
            float_list.extend(position.tolist())
        float_array.data = float_list
        float_array.layout = layout

        return float_array
    
    def run(self):        
        self.jaco_interface.home()
        rospy.spin()

    def make6DofMarker(self, fixed, interaction_mode, show_6dof = False):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.base_frame
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

class SimplePointSimulator():
    """ A simple simulator to show markers for a human and robot """
    def __init__(self, robot_positions, human_positions, jaco_interface=None, repeat=True):
        self.Timer = None
        self.repeat = repeat

        self.simulated_dt = 0.2 #TODO set these better
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
        #rospy.loginfo("human_positions {}".format(self.human_positions))
        #rospy.loginfo("robot_positions {}".format(self.robot_positions))
        #rospy.loginfo("human shape {}".format(np.shape(self.human_positions)))
        #rospy.loginfo("robot shape {}".format(np.shape(self.robot_positions)))
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
        num_sim_wpts = 20 #TODO set this with a parameter server var
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

        rospy.loginfo("sim_pos shape: {}".format(np.shape(sim_pos)))
        playback_pos[:,0] = np.interp(xnew, x, sim_pos[:,0])
        playback_pos[:,1] = np.interp(ynew, y, sim_pos[:,1])
        playback_pos[:,2] = np.interp(znew, z, sim_pos[:,2])
        rospy.loginfo("playback_pos {}".format(playback_pos))

        return playback_pos



if __name__=="__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('jaco_moving_target')
    human_aware_planner = AssertiveRobotPlanner()
    human_aware_planner.run()
    moveit_commander.roscpp_shutdown()