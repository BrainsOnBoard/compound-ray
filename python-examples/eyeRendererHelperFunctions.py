import math
from numpy.ctypeslib import ndpointer
import numpy as np
from ctypes import *

# Configures the renderer's function outputs and inputs:
def configureFunctions(eyeRenderer):
  eyeRenderer.loadGlTFscene.argtypes = [c_char_p]
  eyeRenderer.renderFrame.restype = c_double
  eyeRenderer.getCameraCount.restype = c_size_t
  eyeRenderer.getCurrentCameraIndex.restype = c_size_t
  eyeRenderer.getCurrentCameraName.restype = c_char_p
  eyeRenderer.gotoCameraByName.argtypes = [c_char_p]
  eyeRenderer.gotoCameraByName.restype = c_bool
  eyeRenderer.isCompoundEyeActive.restype = c_bool
  eyeRenderer.getCurrentEyeDataPath.restype = c_char_p

# Updates the render output size while updating the return type of the render pointer
def setRenderSize(eyeRenderer, width, height):
  eyeRenderer.setRenderSize(width, height)
  eyeRenderer.getFramePointer.restype = ndpointer(dtype=c_ubyte, shape = (height, width, 4))

# Sets the samples per ommatidium for the current compound eye in a safer manner,
# i.e. It checks if this is a compound eye, and re-renders to recalculate random seeds
def setSamplesPerOmmatidium(eyeRenderer, samples):
  if(eyeRenderer.isCompoundEyeActive()):
    eyeRenderer.setCurrentEyeSamplesPerOmmatidium(samples)
    eyeRenderer.renderFrame()

# Reads in a given eye file and returns it's information as an array of Ommatidium objects
def readEyeFile(path):
  output = []
  with open(path) as eyeFile:
    for line in eyeFile:
      output.append(getEyeFeatures(line))
  return output
      
class Ommatidium:
  def __init__(self, position, direction, acceptanceAngle, focalpointOffset):
    self.position = position
    self.direction = direction
    self.acceptanceAngle = acceptanceAngle
    self.focalpointOffset = focalpointOffset

  # Returns the solid angle, in steradians, of this ommatidium's cone of vision
  def getSolidAngle(self):
    return (2.0 * math.pi * (1.0-math.cos(self.acceptanceAngle/2.0)))

def getEyeFeatures(line):
  data = [float(n) for n in line.split(" ")]
  position = np.asarray(data[:3])
  direction = np.asarray(data[3:6])
  acceptanceAngle = data[6]
  focalPointOffset = data[7]
  return (Ommatidium(position, direction, acceptanceAngle, focalPointOffset))
