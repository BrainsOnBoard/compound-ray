import math
from numpy.ctypeslib import ndpointer
import numpy as np
from ctypes import *

class Ommatidium:
  def __init__(self, position, direction, acceptanceAngle, focalpointOffset):
    self.position = position
    self.direction = direction
    self.acceptanceAngle = acceptanceAngle
    self.focalpointOffset = focalpointOffset

  # Returns the solid angle, in steradians, of this ommatidium's cone of vision
  def getSolidAngle(self):
    return (2.0 * math.pi * (1.0-math.cos(self.acceptanceAngle/2.0)))

class c_ommatidiumPacket(Structure):
  _fields_ = [
    ("posX", c_float),
    ("posY", c_float),
    ("posZ", c_float),
    ("dirX", c_float),
    ("dirY", c_float),
    ("dirZ", c_float),
    ("acceptanceAngle", c_float),
    ("focalpointOffset", c_float)
  ]

def configureFunctions(eyeRenderer):
  """ Configures the renderer's function outputs and inputs, bar the 'setOmmatidia' method, which is reconfigured depending on the input length in the setOmmatidiaFromX() functions."""
  eyeRenderer.loadGlTFscene.argtypes = [c_char_p]
  eyeRenderer.renderFrame.restype = c_double
  eyeRenderer.getCameraCount.restype = c_size_t
  eyeRenderer.getCurrentCameraIndex.restype = c_size_t
  eyeRenderer.getCurrentCameraName.restype = c_char_p
  eyeRenderer.gotoCameraByName.argtypes = [c_char_p]
  eyeRenderer.gotoCameraByName.restype = c_bool
  eyeRenderer.isCompoundEyeActive.restype = c_bool
  eyeRenderer.getCurrentEyeOmmatidialCount.restype = c_size_t
  eyeRenderer.getCurrentEyeDataPath.restype = c_char_p

def setRenderSize(eyeRenderer, width, height):
  """ Updates the render output size while updating the return type of the render pointer."""
  eyeRenderer.setRenderSize(width, height)
  eyeRenderer.getFramePointer.restype = ndpointer(dtype=c_ubyte, shape = (height, width, 4))

def setSamplesPerOmmatidium(eyeRenderer, samples):
  """ The safe version of setSamplesPerOmmatidiumUnsafe(), checks if the current camera ic compound before proceeding."""
  if(eyeRenderer.isCompoundEyeActive()):
    setSamplesPerOmmatidiumUnsafe(eyeRenderer, samples)

def setSamplesPerOmmatidiumUnsafe(eyeRenderer, samples):
  """ Sets the samples per ommatidium for the current compound eye in a safer manner,
  i.e. It checks if this is a compound eye, and re-renders to recalculate random seeds."""
  eyeRenderer.setCurrentEyeSamplesPerOmmatidium(samples)
  eyeRenderer.renderFrame()

def setOmmatidiaFromPacketListUnsafe(eyeRenderer, packetList):
  """ Sets the current compound eye's ommatidial data from a list of c_ommatidiumPacket objects.
  Notes that this method is different from the other exposed, as the input list must be reconfigured."""
  ommCount = len(packetList)
  #TODO: The below could be a pointer
  c_omm_array_type = c_ommatidiumPacket * ommCount
  eyeRenderer.setOmmatidia.argtypes = [c_omm_array_type, c_size_t]
  c_ommArray = c_omm_array_type(*packetList)
  eyeRenderer.setOmmatidia(c_ommArray, c_size_t(ommCount))

def setOmmatidiaFromPacketList(eyeRenderer, packetList):
  """ The safe version of setOmmatidiaFromPacketListUnsafe, checks that the current eye actually is a compound one."""
  if(eyeRenderer.isCompoundEyeActive()):
    setOmmatidiaFromPacketListUnsafe(eyeRenderer, packetList)

def setOmmatidiaFromOmmatidiumListUnsafe(eyeRenderer, ommList):
  """ Sets the current compound eye's ommatidial data from a list of Ommatidium objects.
  Notes that this method is different from the other exposed, as the input list must be reconfigured."""
  # Convert each Ommatidium to a c_ommatidiumPacket
  packetList = [c_ommatidiumPacket(*[float(n) for n in o.position], *[float(n) for n in o.direction], o.acceptanceAngle, o.focalpointOffset) for o in ommList]
  # Do the rest normally
  setOmmatidiaFromPacketList(eyeRenderer, packetList)

def setOmmatidiaFromOmmatidiumList(eyeRenderer, ommList):
  """ The safe version of setOmmatidiaFromOmmatidiumList, checks that the current eye actually is a compound one."""
  if(eyeRenderer.isCompoundEyeActive()):
    setOmmatidiaFromOmmatidiumListUnsafe(eyeRenderer, ommList)

def readEyeFile(path):
  """ Reads in a given eye file and returns it's information as an array of Ommatidium objects."""
  output = []
  with open(path) as eyeFile:
    for line in eyeFile:
      output.append(_getEyeFeatures(line))
  return output
      

def _getEyeFeatures(line):
  data = [float(n) for n in line.split(" ")]
  position = np.asarray(data[:3])
  direction = np.asarray(data[3:6])
  acceptanceAngle = data[6]
  focalPointOffset = data[7]
  return (Ommatidium(position, direction, acceptanceAngle, focalPointOffset))
