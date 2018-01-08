# Overview
**Background** Path planning finds a trajectory, or set of waypoints for joints, for robotic arm from a start point to a goal. Optimization can be used to find the optimal set of waypoints based upon a cost function. Several open source tools have been developed for this. This project uses an opensource trajectory optimizer called trajopt.

Code and information about trajopt can be found here: http://rll.berkeley.edu/trajopt/doc/sphinx_build/html/

This codebase contains models for human behavior and cost functions to define the action of robots with the intent of affecting the human's actions. By modeling the human behavior properly a robot should be able to get a desired response from a human. Our goal is to show that even with very simple models for a how a human behaves - reasonable interaction behavior can result.

**Example** We showed that if the human is assumed to operate a charged point mass and the robot is given a cost which is higher if the human is moving faster then the robot will pull in front of the human to slow it down - this behavior was not explicitly programmed into the robot.

# Running Expriments
After completing the installation - run the following line to launch all code necissary to run the influence aware planner
```
roslaunch influence_aware_planner.launch
```
## Cost functions
We applied several different costs to effect the human. Trajopt minimizes the cost.
 * **human_speed** The human speed is returned as the cost. Slower speed means less cost.
 * **human_go_first** If the robot and human cross paths then the robot is incentivized if the human goes first.
 * **human_closeness_cost** Cost is incurred based on the distance to the human. The cost is higher when the human and robot are closer.

## Human models
We expiremented with three different models
 * **constant_velocity_model** the simplest model, the humans velocity is maintained throughout the trajectory. This is used as a baseline to compare the other models to.
 * **certainty_based_speed** their are a finite set of goals that the human and robot are both working towards. The robot assumes that the human will move toward the goal that the robot is least likely to go towards based on the robots trajectory thus far. The human moves faster the more certain it is the robot is not going towards that goal.
 * **potential_field_model** Assumes the robot is running potenial field planning, moves like a charged point mass with attraction towards a goal and repulsion from the robot. This model can be tricky to choose hyper parameters for

## Parameters
You can modify parameters in the launch file influence_aware_planner.launch to change experiment settings.
1. Human model parameters
 * **/human_model/dt** the timestep between waypoint, used in simulating the humans actions
 * **/human_model/mass** the mass of the human for use with the potential field model
 * **/human_model/human_avoidance** a hyper parameter which trades off between progress towards a goal and moving away from the robot for use with the potential field model
 * **/human_model/drag** a drag parameter for use with the potential field model, reduces oscillation by providing some "damping"
 * **/human_model/force_max** maximum force that the human will put out
 * **/human_model/certainty_speed_max** maximum speed the human will go under the certainty based speed model
 * **human_model/model_name** name of the model, see Human models for options
 * **/goal_inference/variance** the variance of the gaussian for use with goal inference

2. Cost function parameters
 * **/cost_func/hit_human_penalty** The amount to scale things by for the human closensess cost
 * **/cost_func/eef_link_name** the name of the end effector link
 * **/cost_func/care_about_distance** if the robot and human are further apart than this distance the returned cost is zero for potential field model
 * **/cost_func/cost_name** the name of the cost function, see cost functions for options

## Source Code
* **run_experiment.py** The main code for creating and running experiments
* **simulation.py** A simple simulator to display trajectories using markers on a screen
* **cost_affect_human.py** Contains classes and subclasses for cost functions which try to leverage an effect on the human. These cost functions are passed to trajopt in order to be optimized
* **human_model.py** Contains models for how a human might react to a robot's actions
* **jaco: Interface** to all things robot, exposes methods like kinematics, planning and execution for use on the jaco robot arm. Provides kinematics through Moveit! which has better support for visualization tools like rviz.
* **trajopt_interface.py** the low level interface to trajopt - this predominantly exists to try and hide and isolate openRAVE as much as possible until someone writes it out of Trajopt.
* **marker_wrapper.py** tools for displaying markers in rviz
plotting: plotting tools for plotting test data. 


# Installation
1. Install OpenRAVE....good luck, have fun!
http://openrave.org/
First try the interACT wiki instructions. (requires access to interaACT lab wiki)
You could also try these instructions with an older commit:
https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html

make sure you can run the following in a python shell:
import openravepy

2. Install trajopt
http://rll.berkeley.edu/trajopt/doc/sphinx_build/html/install.html
not all ctests have to pass.
You dont have to install Gurobi.

Make sure you can run 
import trajoptpy


3. Install MoveIt! For detailed instructions see [MoveIt install page](http://moveit.ros.org/install/ "http://moveit.ros.org/install/").  
```
sudo apt-get install ros-indigo-moveit
```

4. Install Trac_IK see [Trac_IK repository](https://bitbucket.org/traclabs/trac_ik "https://bitbucket.org/traclabs/trac_ik").  
```
sudo apt-get install ros-indigo-trac-ik
```

5. install ros-control

```
sudo apt-get install ros-indigo-ros-control ros-indigo-ros-controllers
```

6. Checkout kinova-ros in your catkin workspace  
```
git clone https://github.com/Kinovarobotics/kinova-ros
```

7. Checkout interact-manipulation in your catkin workspace

```
git clone https://github.com/eratner/interact-manipulation.git
```

8. Build. 

```
cd ~/catkin_ws
catkin_make
```