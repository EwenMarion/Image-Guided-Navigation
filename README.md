# Image Guided Navigation

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


