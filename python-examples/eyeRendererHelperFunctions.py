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
  eyeRenderer.setVerbosity.argtypes = [c_bool]
  eyeRenderer.loadGlTFscene.argtypes = [c_char_p]
  eyeRenderer.renderFrame.restype = c_double
  eyeRenderer.getCameraCount.restype = c_size_t
  eyeRenderer.getCurrentCameraIndex.restype = c_size_t
  eyeRenderer.getCurrentCameraName.restype = c_char_p
  eyeRenderer.gotoCameraByName.argtypes = [c_char_p]
  eyeRenderer.gotoCameraByName.restype = c_bool
  eyeRenderer.setCameraPosition.argtypes = [c_float]*3
  eyeRenderer.setCameraLocalSpace.argtypes = [c_float]*9
  eyeRenderer.rotateCameraAround.argtypes = [c_float]*4
  eyeRenderer.rotateCameraLocallyAround.argtypes = [c_float]*4
  eyeRenderer.translateCamera.argtypes = [c_float]*3
  eyeRenderer.translateCameraLocally.argtypes = [c_float]*3
  eyeRenderer.isCompoundEyeActive.restype = c_bool
  eyeRenderer.getCurrentEyeOmmatidialCount.restype = c_size_t
  eyeRenderer.getCurrentEyeDataPath.restype = c_char_p
  eyeRenderer.setCurrentEyeShaderName.argtypes = [c_char_p]
  eyeRenderer.setCameraPose.argtypes = [c_float]*6

def setCameraLocalSpace(eyeRenderer, npMatrix):
  newX = npMatrix[:,0]
  newY = npMatrix[:,1]
  newZ = npMatrix[:,2]
  #eyeRenderer.setCameraLocalSpace(newX[0],newX[1],newX[2], newY[0],newY[1],newY[2], newZ[0],newZ[1],newZ[2],)
  eyeRenderer.setCameraLocalSpace(*newX, *newY, *newZ)

def setRenderSize(eyeRenderer, width, height):
  """ Updates the render output size while updating the return type of the render pointer."""
  eyeRenderer.setRenderSize(width, height)
  eyeRenderer.getFramePointer.restype = ndpointer(dtype=c_ubyte, shape = (height, width, 4))

def setOmmatidiaFromPacketList(eyeRenderer, packetList):
  """ Sets the current compound eye's ommatidial data from a list of c_ommatidiumPacket objects.
  Notes that this method is different from the other exposed, as the input list must be reconfigured."""
  ommCount = len(packetList)
  #TODO: The below could be a pointer
  c_omm_array_type = c_ommatidiumPacket * ommCount
  eyeRenderer.setOmmatidia.argtypes = [c_omm_array_type, c_size_t]
  c_ommArray = c_omm_array_type(*packetList)
  eyeRenderer.setOmmatidia(c_ommArray, c_size_t(ommCount))

def setOmmatidiaFromOmmatidiumList(eyeRenderer, ommList):
  """ Sets the current compound eye's ommatidial data from a list of Ommatidium objects.
  Notes that this method is different from the other exposed, as the input list must be reconfigured."""
  # Convert each Ommatidium to a c_ommatidiumPacket
  packetList = [c_ommatidiumPacket(*[float(n) for n in o.position], *[float(n) for n in o.direction], o.acceptanceAngle, o.focalpointOffset) for o in ommList]
  # Do the rest normally
  setOmmatidiaFromPacketList(eyeRenderer, packetList)

def readEyeFile(path):
  """ Reads in a given eye file and returns it's information as an array of Ommatidium objects."""
  output = []
  with open(path) as eyeFile:
    for line in eyeFile:
      output.append(_getEyeFeatures(line))
  return output

def decodeProjectionMapID(RGBAquadlet):
  """ Given the RGBA quadlet from a pixel which is encoded as an ID using an "_ids" shader."""
  r = RGBAquadlet[0] << 24 # Red
  g = RGBAquadlet[1] << 16 # Green
  b = RGBAquadlet[2] <<  8 # Blue
  a = RGBAquadlet[3]       # Alpha
  idOut = r | g | b | a
  return(idOut)

def getProjectionImageUsingMap(vector, idMap, pjWidth, pjHeight):
  """ Uses an id map generated using an "_ids" shader, and re-projects the vector outputs to their correct locations on the projection map. vector components must be between 0 and 255 inclusive."""
  output = np.zeros((pjWidth, pjHeight), dtype=np.uint8)
  for x in range(pjWidth):
    for y in range(pjHeight):
      pixelId = decodeProjectionMapID(idMap[y,x,:])
      output[y,x] = int(vector[pixelId])
  return output

def _getEyeFeatures(line):
  data = [float(n) for n in line.split(" ")]
  position = np.asarray(data[:3])
  direction = np.asarray(data[3:6])
  acceptanceAngle = data[6]
  focalPointOffset = data[7]
  return (Ommatidium(position, direction, acceptanceAngle, focalPointOffset))
