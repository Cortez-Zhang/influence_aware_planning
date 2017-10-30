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
