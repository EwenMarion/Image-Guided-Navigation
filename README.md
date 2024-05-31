# 3DSlicer

# Overview:
  the aim of this slicer extension is to find the optimal safest path between a plethora of entry and target points by filtering for length of the trajectory, checking for collisions with critical structures and maximising the distance to a critical structure.

# Installation:
  1. open the extension wizard in the development tools tab
  2. use the "select extension" option
  3. select the PathPlanning folder containing the extension

# How it works:
  1. Setting variables
    The Volumes and Markups Fiducials are assigned following the inputs in the GUI
  
  2. Chosing the optimal entry and target point pair
    For every entry and target point combination, iterate over multiple filters:
    - A length filter that measures distance between 2 points and verifies if it is under the set threshold
    - A collision filter that checks if the line between the points intersect with a critical structure
    - A distance measurement that measures the distance between the line and the critical structure for all lines that do not collide
    
  The best path is one that is shorter than the length threshold,
  does not collide with critical structures
  and stays the furthest away from a critical structure

# Usage:
  import the following data into slicer, making sure the data type is correct:
  1. inputTargetVolume: vtkMRMLLabelMapVolumeNode
  2. inputCriticalVolume: vtkMRMLLabelMapVolumeNode
  3. inputEntryFiducials: vtkMRMLMarkupsFiducialNode
  4. inputTargetFiducials: vtkMRMLMarkupsFiducialNode

  make sure to select LabelMap in more options for the volume nodes

  select the wanted input nodes and maximum length threshold using the GUI
  select "create new points list" in the outputs tab to generate the results
  press the apply button to run the calculations

# IGT Link
  download OpenIGTLink from the extensions manager
  In the OpenIGTLinkIF tab, click the + to add a connector and set the status to active
  in the I/O configuration, add the best_points vtkMRMLMarkupsFiducialNode to the OUT section.

  In ROS, open a terminal and source setup.bash
  change the directory to catkin_ws/src/ROS-IGTL-Bridge/launch and run roslaunch ros_igtl_bridge bridge.launch to set up a connection to 3D Slicer

  In a new terminal, run rostopic echo /IGTL_POINT_IN to create a listener node and be able to receive the point data
  Go back to slicer and press the send button, the point information should appear in your ros terminal under the echo

# ROS

# Robot URDF
  The urdf defines the shape of the model for the virtually simulated robot.
  The robot is comprised of 3 2DOF joints connecting 3 links with a sphere marking the end effector at the tip
  The urdf is used as the base for moveit package which will add collision detection and inverse kinematics calculations to allow the robot to be simulated

# Moveit!
  open Moveit! using by typing "roslaunch moveit_setup_assistant setup_assistant.launch" in the command window
  create a newpackage and select your URDF file path
  in the self-collisions tab, click generate collision matrix
  create a new planning group with all the joints and select kdl_kinematics_plugin/KDLKinematicsPlugin as the kinematic solver and LazyPRMstar as the group default planner
  go to the configure tab and generate the package

# Rviz
  type roslaunch "moveit package name" demo.launch to open Rviz
  select the wanted start and goal states
  click plan and execute to see the robot move to its target destination