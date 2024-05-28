from importlib.metadata import entry_points
import logging
import os
from typing import Annotated, Optional

import vtk
import numpy as np
import SimpleITK as sitk
import sitkUtils
import time

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (parameterNodeWrapper, WithinRange)

from slicer import vtkMRMLLabelMapVolumeNode, vtkMRMLMarkupsFiducialNode, vtkMRMLUnitNode

#
# PathPlanning
#


class PathPlanning(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("PathPlanning")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ewen Marion - KCL"]
        self.parent.helpText = _("""
This is the start of the path planning script with some helpers already implemented
""")
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.  Rachel Sparks has modified this, 
for part of Image-guide Navigation for Robotics taught through King's College London.
""")


# PathPlanningParameterNode
#


@parameterNodeWrapper
class PathPlanningParameterNode:
    """
    The parameters needed by module.

    inputTargetVolume - The label map the trajectory must be inside
    inputCriticalVolume - The label map the trajectory avoid
    inputEntryFiducials - Fiducials cotaining potential target points
    inputTargetFiducials - Fiducials containing potential entry points
    lengthThreshold - The value above which to exclude trajectories
    outputFiducials - Fiducials containing output points of target and entry pairs
    """

    inputTargetVolume: vtkMRMLLabelMapVolumeNode
    inputCriticalVolume: vtkMRMLLabelMapVolumeNode
    inputEntryFiducials: vtkMRMLMarkupsFiducialNode
    inputTargetFiducials: vtkMRMLMarkupsFiducialNode
    lengthThreshold: Annotated[float, WithinRange(0, 500)] = 100
    outputFiducials: vtkMRMLMarkupsFiducialNode



#
# PathPlanningWidget
#


class PathPlanningWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/PathPlanning.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = PathPlanningLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.clicked.connect(self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputTargetVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLabelMapVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputTargetVolume = firstVolumeNode

        if not self._parameterNode.inputCriticalVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLabelMapVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputCriticalVolume = firstVolumeNode

        if not self._parameterNode.inputTargetFiducials:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
            if firstVolumeNode:
                self._parameterNode.inputTargetFiducials = firstVolumeNode

        if not self._parameterNode.inputEntryFiducials:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
            if firstVolumeNode:
                self._parameterNode.inputEntryFiducials = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[PathPlanningParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputTargetVolume and self._parameterNode.inputCriticalVolume and self._parameterNode.inputEntryFiducials and self._parameterNode.inputTargetFiducials:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select all input nodes")
            self.ui.applyButton.enabled = False
    
    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        # First initialise the logic
        self.logic = PathPlanningLogic()
        # Set class parameters
        self.logic.SetEntryPoints(self.ui.inputEntryFiducialSelector.currentNode())
        self.logic.SetTargetPoints(self.ui.inputTargetFiducialSelector.currentNode())
        self.logic.SetOutputPoints(self.ui.outputFiducialSelector.currentNode())
        self.logic.SetInputTargetImage(self.ui.inputTargetVolumeSelector.currentNode())
        self.logic.SetInputCriticalVolume(self.ui.inputCriticalVolumeSelector.currentNode())
        self.logic.SetLengthThreshold(self.ui.lengthSliderWidget.value)
        # finally try to run the code. Return false if the code did not run properly
        complete = self.logic.run()

        if complete:
            criticalVolume = self.ui.inputCriticalVolumeSelector.currentNode()
            pointPicker = PickPointsMatrix()
            pointPicker.GetLines(self.logic.myEntries, self.logic.myOutputs, "lineNodes", criticalVolume, self.logic.lengthThreshold)

        if self.ui.inputTargetFiducialSelector.currentNode():
            self.ui.inputTargetFiducialSelector.currentNode().SetDisplayVisibility(False)

        if not complete:
            print('I encountered an error')

#
# PathPlanningLogic
#

class PathPlanningLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return PathPlanningParameterNode(super().getParameterNode())

    def SetEntryPoints(self, entryNode):
        self.myEntries = entryNode

    def SetTargetPoints(self, targetNode):
        self.myTargets = targetNode

    def SetInputTargetImage(self, imageNode):
        if (self.hasImageData(imageNode)):
            self.myTargetImage = imageNode

    def SetInputCriticalVolume(self, criticalVolumeNode):
        self.myCriticalVolume = criticalVolumeNode

    def SetOutputPoints(self, outputNode):
        self.myOutputs = outputNode

    def SetLengthThreshold(self, lengthThreshold):
        self.lengthThreshold = lengthThreshold

    def hasImageData(self, volumeNode):
        if not volumeNode:
            logging.debug('hasImageData failed: no volume node')
            return False
        if volumeNode.GetImageData() is None:
            logging.debug('hasImageData failed: no image data in volume node')
            return False
        return True

    def isValidInputOutputData(self, inputTargetVolumeNode, inputCriticalVolumeNode, inputTargetFiducialsNode, inputEntryFiducialsNode, outputFiducialsNode):
        """Validates if the output is not the same as input
        """
        if not inputTargetVolumeNode:
            logging.debug('isValidInputOutputData failed: no input target volume node defined')
            return False
        if not inputCriticalVolumeNode:
            logging.debug('isValidInputOutputData failed: no input critical volume node defined')
            return False
        if not inputTargetFiducialsNode:
            logging.debug('isValidInputOutputData failed: no input target fiducials node defined')
            return False
        if not inputEntryFiducialsNode:
            logging.debug('isValidInputOutputData failed: no input entry fiducials node defined')
            return False
        if not outputFiducialsNode:
            logging.debug('isValidInputOutputData failed: no output fiducials node defined')
            return False
        if inputTargetFiducialsNode.GetID() == outputFiducialsNode.GetID():
            logging.debug('isValidInputOutputData failed: input and output fiducial nodes are the same. Create a new output to avoid this error.')
            return False
        return True

    def run(self):
        """
        Run the path planning algorithm.
        """

        if not self.isValidInputOutputData(self.myTargetImage, self.myCriticalVolume, self.myTargets, self.myEntries, self.myOutputs):
            slicer.util.errorDisplay('Not all inputs are set.')
            return False
        if not self.hasImageData(self.myTargetImage):
            raise ValueError("Input target volume is not appropriately defined.")

        import time

        startTime = time.time()        
        logging.info("Processing started")
        
        #running one of the two methods for picking suitable target points as shown below
        pointPicker = PickPointsMatrix()
        pointPicker.run(self.myTargetImage,  self.myTargets, self.myOutputs)
        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")
        return True


class PickPointsMatrix(ScriptedLoadableModuleLogic): 
    def run(self, inputVolume, inputFiducials, outputFiducials):
        outputFiducials.RemoveAllControlPoints()
        mat = vtk.vtkMatrix4x4()
        inputVolume.GetRASToIJKMatrix(mat)
        transform = vtk.vtkTransform()
        transform.SetMatrix(mat)

        for x in range(inputFiducials.GetNumberOfControlPoints()):
            pos = [0, 0, 0]
            inputFiducials.GetNthControlPointPosition(x, pos)
            # get index from position using our transformation
            ind = transform.TransformPoint(pos)

            # get pixel using that index
            pixelValue = inputVolume.GetImageData().GetScalarComponentAsDouble(int(ind[0]), int(ind[1]), int(ind[2]), 0)  # looks like it needs 4 ints -- our x,y,z index and a component index (which is 0)
            if pixelValue == 1:
                outputFiducials.AddControlPoint(pos[0], pos[1], pos[2])

    def GetLines(self, inputEntryFiducials, outputFiducials, groupName, CriticalVolume, lengthThreshold):
    # Initialise  the transformation matrix and the imageData as they remain constant
        mat = vtk.vtkMatrix4x4()
        CriticalVolume.GetRASToIJKMatrix(mat)
        imageData = CriticalVolume.GetImageData()

        numberOfEntryPoints = inputEntryFiducials.GetNumberOfControlPoints()
        numberOfTargetPoints = outputFiducials.GetNumberOfControlPoints()

        print("number of entry points: ", numberOfEntryPoints)
        print("number of target points: ", numberOfTargetPoints)

        distanceMap = self.GenerateDistanceMap(CriticalVolume)
        
        safestDistance = 0

        startTime = time.time()

        for entryIndex in range(numberOfEntryPoints):
            entryPointRAS = [0, 0, 0]
            inputEntryFiducials.GetNthControlPointPosition(entryIndex, entryPointRAS)
            for targetIndex in range(numberOfTargetPoints):
                targetPointRAS = [0, 0, 0]
                outputFiducials.GetNthControlPointPosition(targetIndex, targetPointRAS)

                pathLength = np.linalg.norm(np.array(targetPointRAS) - np.array(entryPointRAS))
                if pathLength > lengthThreshold:
                    print("Line too long")
                    continue
                print(f"Line length = {pathLength:.2f}")

                if self.CollisionDetection(CriticalVolume, entryPointRAS, targetPointRAS, mat):
                    print("collision detected")
                    continue
                print("no collision with critical structure")

                critDistance = self.DistanceToCriticalVolume(entryPointRAS, targetPointRAS, mat, distanceMap)
                print(f"min distance =  {critDistance:.2f}")

                if critDistance > safestDistance:
                    safestDistance = critDistance
                    bestEntryIndex = entryIndex
                    bestTargetIndex = targetIndex
                    bestEntry = entryPointRAS
                    bestTarget = targetPointRAS
                    print("new best line")
                
                print("")

        print(f"The best trajectory is from entry point {bestEntryIndex} to target point {bestTargetIndex}")
        print(f"entry point location {bestEntry}")
        print(f"best target point location {bestTarget}")
        print(f"distance to critical structure {safestDistance}")

        lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        lineNode.RemoveAllControlPoints()
        lineNode.AddControlPoint(bestEntry)
        lineNode.AddControlPoint(bestTarget)
        lineNode.SetAttribute("LineGroup", groupName)
        slicer.mrmlScene.AddNode(lineNode)

        stopTime = time.time()
        print(f"runtime = {stopTime - startTime}")


    def CollisionDetection(self, volume, entryPoint, targetPoint, mat):
        sitkImage = sitkUtils.PullVolumeFromSlicer(volume.GetID())
        sitkImage = sitk.Cast(sitkImage, sitk.sitkUInt8)
        sitkImage = sitk.GetArrayFromImage(sitkImage)

        IJKentryPoint = [0, 0, 0, 1]
        IJKtargetPoint = [0, 0, 0, 1]
        
        # Apply the transformation matrix to entryPoint and targetPoint
        entryPointHomogeneous = entryPoint + [1]
        targetPointHomogeneous = targetPoint + [1]
        
        mat.MultiplyPoint(entryPointHomogeneous, IJKentryPoint)
        mat.MultiplyPoint(targetPointHomogeneous, IJKtargetPoint)

        # Remove the homogeneous coordinate
        IJKentryPoint = IJKentryPoint[:3]
        IJKtargetPoint = IJKtargetPoint[:3]

        for samplePoint in (np.linspace(0, 1, num=100)[:, None] * (np.subtract(IJKtargetPoint, IJKentryPoint) + IJKentryPoint)):
            # Convert samplePoint to integer IJK coordinates
            samplePointIJK = np.array(samplePoint).astype(int)
            if np.any(sitkImage < 0) or np.any(samplePointIJK >= sitkImage.shape):
                continue
            if sitkImage[samplePointIJK[2], samplePointIJK[1], samplePointIJK[0]] != 0:
                return True
            return False



    def GenerateDistanceMap(self, volume):
        sitkImage = sitkUtils.PullVolumeFromSlicer(volume.GetID())
        sitkImage = sitk.Cast(sitkImage, sitk.sitkUInt8)  # Convert to a supported pixel type
        distanceFilter = sitk.DanielssonDistanceMapImageFilter()
        distanceMap = distanceFilter.Execute(sitkImage)
        return sitk.GetArrayFromImage(distanceMap)
                
        
    def DistanceToCriticalVolume(self, entryPoint, targetPoint, mat, distanceMap):
        IJKentryPoint = [0, 0, 0, 1]
        IJKtargetPoint = [0, 0, 0, 1]
        
        # Apply the transformation matrix to entryPoint and targetPoint
        entryPointHomogeneous = entryPoint + [1]
        targetPointHomogeneous = targetPoint + [1]
        
        mat.MultiplyPoint(entryPointHomogeneous, IJKentryPoint)
        mat.MultiplyPoint(targetPointHomogeneous, IJKtargetPoint)

        # Remove the homogeneous coordinate
        IJKentryPoint = IJKentryPoint[:3]
        IJKtargetPoint = IJKtargetPoint[:3]

        minDistanceToCriticalStructure = np.inf

        # Generate sample points between entryPoint and targetPoint
        for samplePoint in (np.linspace(0, 1, num=100)[:, None] * (np.subtract(IJKtargetPoint, IJKentryPoint) + IJKentryPoint)):
            # Convert samplePoint to integer IJK coordinates
            samplePointIJK = np.array(samplePoint).astype(int)
            samplePointDistance = distanceMap[samplePointIJK[2], samplePointIJK[1], samplePointIJK[0]]
            if samplePointDistance < minDistanceToCriticalStructure:
                minDistanceToCriticalStructure = samplePointDistance
        return minDistanceToCriticalStructure

            




#
# PathPlanningTest
#
class PathPlanningTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

        self.delayDisplay("Starting load data test")
        # path to the test set given to class. Alter to run on your own system
        path = '/Users/takrim/Library/Mobile Documents/com~apple~CloudDocs/University/MSc Healthcare Technologies/Image Guided Navigation/Week 2/Tutorial/Week23/TestSet'
        isLoaded = slicer.util.loadVolume(path + '/r_hippoTest.nii.gz')
        if (not isLoaded):
            self.delayDisplay('Unable to load ' + path + '/r_hippoTest.nii.gz')

        self.delayDisplay('Test passed! All data loaded correctly')

    def test_PathPlanningTestOutsidePoint(self):
        """Here I give a point I know should be outside for hippocampus.
           Hence I expect the return markups fiducial node to be empty.
        """

        self.delayDisplay("Starting the test")
        self.delayDisplay("Starting test points outside mask.")
        #
        # get our mask image node
        mask = slicer.util.getNode('r_hippoTest')

        # I am going to hard code two points -- both of which I know are not in my mask
        outsidePoints = slicer.vtkMRMLMarkupsFiducialNode()
        outsidePoints.AddControlPoint(-1, -1, -1)  # this is outside of our image bounds
        cornerPoint = mask.GetImageData().GetOrigin()
        outsidePoints.AddControlPoint(cornerPoint[0], cornerPoint[1], cornerPoint[2])  # we know our corner pixel is no 1

        #run our class
        returnedPoints = slicer.vtkMRMLMarkupsFiducialNode()
        PickPointsMatrix().run(mask, outsidePoints, returnedPoints)

        # check if we have any returned fiducials -- this should be empty
        if (returnedPoints.GetNumberOfControlPoints() > 0):
            self.delayDisplay('Test failed. There are ' + str(returnedPoints.GetNumberOfControlPoints()) + ' return points.')
            return

        self.delayDisplay('Test passed! No points were returned.')
