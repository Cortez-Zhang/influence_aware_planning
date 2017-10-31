# Running Jaco Arm in Gazebo/RViz

1. Launch the Jaco 7 DOF arm in Gazebo:
```
  roslaunch kinova_gazebo robot_launch.launch kinova_robotType:=j2s7s300 headless:=true gui:=false is7dof:=true
```
2. Launch MoveIt! for the arm:
```
  roslaunch interact_manipulation jaco_gazebo.launch
```
3. Run RViz (for visualization):
```
  rosrun rviz rviz
```
4. First, you may want to load a pre-configured RViz configuration: in RViz, File -> Open Config, choose config/jaco.rviz. Then, run the basic script:
```
  rosrun interact_manipulation jaco.py /joint_states:=/j2s7s300/joint_states
```

# Installation

1. Install OpenRAVE....good luck, have fun!
http://openrave.org/
First try the interACT wiki instructions.
You could also try these instructions with an older commit:
https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html

make sure you can run
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