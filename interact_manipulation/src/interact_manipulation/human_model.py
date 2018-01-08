import math
import rospy
import copy
import numpy as np
import util
import marker_wrapper


class HumanState:
    """Stores the state of a human
        Params
        ---
        position: Human position (3,) numpy array
        velocity: Human velocity (3,) numpy array
    """
    def __init__(self, position, velocity):
        """

        """
        self.position = position
        self.velocity = velocity

    @property
    def position(self):
        #TODO add assertion here
        return self.position
    
    @position.setter
    def position(self, pos):
        #TODO add assertion here
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

class ModelFactory(object):
    @staticmethod
    def factory(model_name, human_start_state):
        goal1 = np.array([-0.4,-0.1,0.538])
        goal2 = np.array([-0.2,-0.2,0.538])

        if model_name == "certainty_based_speed":
            goals = [goal1, goal2]
            goal_inference = GoalInference(goals)
            human_model = CertaintyBasedSpeedModel(human_start_state, goals, goal_inference=goal_inference)
        elif model_name == "constant_velocity_model":
            human_model = ConstantVelocityModel(human_start_state)
        elif model_name == "potential_field_model":
            human_model = PotentialFieldModel(human_start_state, goal1)
        else:
            err = "No human model object exists with model name {}".format(model_name)
            rospy.logerr(err)
            raise ValueError(err)
        return human_model

class HumanModel(object):
    """ Simulates a human for use in path planning (trajopt)
        Params
        ---
        start_state: starting HumanState object
        goals: A list of (3,) numpy arrays, known goal_positions of the human
        simulation method: The method in which to simulate the human 
        e.g. constant velocity, point_mass, certainty_based_speed
        dt: the fixed time between waypoints (default 0.2)
    """
    def __init__(self, start_state) :
        self.start_state = copy.deepcopy(start_state)
        
        #specify which parameters I would like to use
        params = ["dt","mass","human_avoidance","drag","force_max","certainty_speed_max"]
        namespace = "/human_model/"
        self.params = util.set_params(params,namespace)
        
        #list to keep track of human positions for displaying
        self.human_positions = []
        self.current_state = copy.deepcopy(start_state)
        self.human_positions.append(start_state.position)

        marker_wrapper.show_position_marker(label="human \n start\n\n", position = start_state.position, ident=1, color=(1,0,0))

    def reset_model(self):
        """ Reset the model to prepare for another forward simulation
        """            
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
        prev_eef_pos = eef_positions[0]
        for eef_pos in eef_positions:
            self.advance_model(eef_pos, prev_eef_pos)
            prev_eef_pos = eef_pos.copy()

        return self.human_positions
    
    def advance_model(self,eef_position, prev_eef_pos):
        """advance the model one step based upon some method"""
        pass
        #TODO add util not defined

class CertaintyBasedSpeedModel(HumanModel):
    def __init__(self, start_state, goals, goal_inference):
        super(CertaintyBasedSpeedModel, self).__init__(start_state)
        self.goals = goals
        self.goal_inference = goal_inference
        for i, goal in enumerate(goals):
           marker_wrapper.show_position_marker(label="human \n goal\n\n", position= goal, ident=10+i, color=(0,1,0))
    
    def reset_model(self):
        super(CertaintyBasedSpeedModel, self).reset_model()
        self.goal_inference.reset()

    def advance_model(self, eef_position, prev_eef_pos):
        """ Human moves at a speed proportional to belief over robot goals
            The human will move to the goal the robot is not going towards
            the human moves faster if it is more certain that is the goal
            Param
            ---
            eef_position: a (3,) numpy array with xyz position of robot end effector
        """     
        curr_pos = self.current_state.position
        #check to make sure we dont jump over the goal and oscilate forever
        if all(np.linalg.norm(curr_pos - goal)>0.05 for goal in self.goals):
            self.goal_inference.update(eef_position,prev_eef_pos)
            b = self.goal_inference.current_beliefs

            #speed is faster if we are more certain the robot isnt going towards a goal
            speed = self.params["certainty_speed_max"]*(1-(min(b)*len(self.goals)))
            human_goal = b.index(min(b))
            goal_dir = util.direction(curr_pos,self.goals[human_goal]) #humans goal direction
            
            next_vel = speed*goal_dir
            next_pos = next_vel*self.params["dt"]+curr_pos
            
            self.current_state.position = next_pos
            self.current_state.velocity = next_vel
            
        else:
            next_pos = self.current_state.position
            next_vel = self.current_state.velocity
            #self.human_velocities
        self.human_velocities.append(next_vel.copy())
        self.human_positions.append(next_pos.copy())

class ConstantVelocityModel(HumanModel):
    #TODO get rid of advance model and vectorize all in get_human_position
    def advance_model(self, eef_position, prev_eef_pos):
        """ Evolve the human state forward using a constant velocity assumption
        """
        curr_pos = self.current_state.position
        next_pos = curr_pos + self.current_state.velocity*self.params["dt"]
        self.current_state.position = next_pos
        self.human_positions.append(next_pos)

class PotentialFieldModel(HumanModel):
    def __init__(self, start_state, goal):
        super().__init__(start_state)
        self.goal = goals[0]
        #the tuning currently only works with one goal
    
    def advance_model(self, eef_position, prev_eef_pos):
        """ Evolve the human state forward using a point mass model
            We assume the human will move away from the robot and towards its goal
        """
        curr_pos = self.current_state.position
        curr_vel = self.current_state.velocity

        F_repulse = -1*self.params["human_avoidance"]*self.potential_field(eef_position,curr_pos)
        
        F_attract = 0.0
        for goal in goals:
            F_attract+= (1-self.params["human_avoidance"])*self.potential_field(self.goal,curr_pos)
        
        acc = (F_attract+F_repulse)*self.params["mass"]

        next_vel = curr_vel+acc*self.params["dt"]
        next_pos = 0.5*acc*self.params["dt"]**2+curr_vel*self.params["dt"]+curr_pos

        self.current_state.position = next_pos
        self.current_state.velocity = next_vel

        self.human_velocities.append(next_vel)
        self.human_positions.append(next_pos)

    def potential_field(self, obstacle, curr_pos):
        """ Calculate distance penalty for obstacles
            Params
            ---
            obstacle: a (3,) numpy array with position of obstacle or goal
            curr_pos: a (3,) numpy array with position of human
        """ 
        epsilon = self.params["force_max"]

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
    def __init__(self, goals, current_beliefs = [0.5,0.5]):
        self.beliefs_over_time = [] #a list of lists, each belief in time
        self.goals = goals
        self.variance = rospy.get_param("/goal_inference/variance")
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
        #find the direction of all the goals
        goal_dirs = [util.direction(eef_pos,goal) for goal in self.goals]
        
        # the movement an agent takes from one step to the next provides evidence for the 
        # direction which is used to update beliefs
        interaction_dir = util.direction(prev_eef_pos,eef_pos)
        
        #update the beliefs based on the new information from the movement the human has made (interaction dir)
        beliefs = np.array([b*self.gaussian(goal_dir)(interaction_dir) for (b, goal_dir) in zip(self.current_beliefs, goal_dirs)])
        norm_belief = util.normalize(beliefs)
        
        #update the beliefs and store the belief history for debug and analysis
        self.beliefs_over_time.append(norm_belief)
        self.current_beliefs = norm_belief.tolist()
