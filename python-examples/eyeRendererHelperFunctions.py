import math
from numpy.ctypeslib import ndpointer
import numpy as np
from ctypes import *

class c_float3(Structure):
  _fields_ = [('x', c_float),
              ('y', c_float),
              ('z', c_float)]
  def toNumpy(self):
    return np.asarray([self.x, self.y, self.z])

class Ommatidium:
  def __init__(self, position, direction, acceptanceAngle, focalpointOffset):
    self.position = position
    self.direction = direction
    self.acceptanceAngle = acceptanceAngle
    self.focalpointOffset = focalpointOffset

  def getSolidAngle(self):
    """ Returns the solid angle, in steradians, of this ommatidium's cone of vision. """
    return (2.0 * math.pi * (1.0-math.cos(self.acceptanceAngle/2.0)))

  def copy(self):
    """ Returns a deep copy."""
    return Ommatidium(self.position.copy(), self.direction.copy(), self.acceptanceAngle, self.focalpointOffset)

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
  eyeRenderer.getCameraPosition.restype = ndpointer(dtype=c_double, shape = (3,1))
  eyeRenderer.setCameraLocalSpace.argtypes = [c_float]*9
  eyeRenderer.rotateCameraAround.argtypes = [c_float]*4 # Angle, x, y, z
  eyeRenderer.rotateCameraLocallyAround.argtypes = [c_float]*4 # Angle, x, y, z
  eyeRenderer.translateCamera.argtypes = [c_float]*3
  eyeRenderer.translateCameraLocally.argtypes = [c_float]*3
  eyeRenderer.isCompoundEyeActive.restype = c_bool
  eyeRenderer.setCurrentEyeSamplesPerOmmatidium.argtypes = [c_int]
  eyeRenderer.getCurrentEyeSamplesPerOmmatidium.restype = c_int
  eyeRenderer.changeCurrentEyeSamplesPerOmmatidiumBy.argtypes = [c_int]
  eyeRenderer.getCurrentEyeOmmatidialCount.restype = c_size_t
  eyeRenderer.getCurrentEyeDataPath.restype = c_char_p
  eyeRenderer.setCurrentEyeShaderName.argtypes = [c_char_p]
  eyeRenderer.setCameraPose.argtypes = [c_float]*6 # pos x, y, z, rotation about x, y, z
  eyeRenderer.saveFrameAs.argtypes = [c_char_p]
  eyeRenderer.getGeometryMaxBounds.argtypes = [c_char_p]
  eyeRenderer.getGeometryMaxBounds.restype = c_float3
  eyeRenderer.getGeometryMinBounds.argtypes = [c_char_p]
  eyeRenderer.getGeometryMinBounds.restype = c_float3
  eyeRenderer.isInsideHitGeometry.restype = c_bool
  eyeRenderer.isInsideHitGeometry.argtypes = [c_float, c_float, c_float, c_char_p]

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
  Note that this method is different from the other exposed, as the input list must be reconfigured."""
  ommCount = len(packetList)
  #TODO: The below could be a pointer
  c_omm_array_type = c_ommatidiumPacket * ommCount
  eyeRenderer.setOmmatidia.argtypes = [c_omm_array_type, c_size_t]
  c_ommArray = c_omm_array_type(*packetList)
  eyeRenderer.setOmmatidia(c_ommArray, c_size_t(ommCount))

def setOmmatidiaFromOmmatidiumList(eyeRenderer, ommList):
  """ Sets the current compound eye's ommatidial data from a list of Ommatidium objects.
  Note that this method is different from the other exposed, as the input list must be reconfigured."""
  # Convert each Ommatidium to a c_ommatidiumPacket
  packetList = [c_ommatidiumPacket(*[float(n) for n in o.position], *[float(n) for n in o.direction], o.acceptanceAngle, o.focalpointOffset) for o in ommList]
  # Do the rest normally
  setOmmatidiaFromPacketList(eyeRenderer, packetList)

def gotoFirstCompoundEye(eyeRenderer):
  """ Searches for a compound eye in the current scene and goes to it. Raises exception if infeasible. """
  foundCompound = False
  camCount = eyeRenderer.getCameraCount()
  for i in range(camCount):
    eyeRenderer.gotoCamera(int(i))
    if eyeRenderer.isCompoundEyeActive():
      foundCompound = True
      print("Found compound eye:", eyeRenderer.getCurrentCameraName())
      print("\twith compound data at:", eyeRenderer.getCurrentEyeDataPath())
      print("\twith this many ommatidia:", eyeRenderer.getCurrentEyeOmmatidialCount())
      break
  if not foundCompound:
    raise Exception("Error: Could not find compound eye in provided GlTF scene.")

def gotoFirstRegularCamera(eyeRenderer):
  """ Searches for a regular camera (panoramic, pinhole, or orthogonal) in the current scene and goes to it. Raises exception if infeasible. """
  foundCamera = False
  camCount = eyeRenderer.getCameraCount()
  for i in range(camCount):
    eyeRenderer.gotoCamera(int(i))
    if not eyeRenderer.isCompoundEyeActive():
      foundCamera = True
      print("Found regular camera:", eyeRenderer.getCurrentCameraName())
      break
  if not foundCamera:
    raise Exception("Error: Could not find compound eye in provided GlTF scene.")

def readEyeFile(path):
  """ Reads in a given eye file and returns it's information as an array of Ommatidium objects."""
  output = []
  with open(path) as eyeFile:
    for line in eyeFile:
      output.append(_getEyeFeatures(line))
  return output

def saveEyeFile(path, omms):
  """ Saves a list of Ommatidium objects as a .eye file."""
  with open(path, "w") as eyeFile:
    for omm in omms:
      eyeFile.write("{:0.10f} {:0.10f} {:0.10f} {:0.10f} {:0.10f} {:0.10f} {:0.10f} {:0.10f}\n".format(
                    omm.position[0],
                    omm.position[1],
                    omm.position[2],
                    omm.direction[0],
                    omm.direction[1],
                    omm.direction[2],
                    omm.acceptanceAngle,
                    omm.focalpointOffset))

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

def getIcoOmmatidia():
  """Returns an ommatidial array based on the points in an icosphere, so they're equidistant.
  Each ommatidium has an acceptance angle of 1 steradian."""
  ## First generate the points of the icosphere
  icoPoints = []
  icoPoints.append([0,1,0]) # Top Point
  angles = [0.4 * math.pi * i for i in range(5)]
  offsetAngle = math.atan(0.5)
  # Add the upper 5 points
  planarDistance = math.cos(offsetAngle)
  verticalOffset = math.sin(offsetAngle)
  icoPoints = icoPoints + [[math.cos(a)*planarDistance, verticalOffset, math.sin(a)*planarDistance] for a in angles]
  # Add the lower 5 points
  angles = [a + 0.2 * math.pi for a in angles]
  verticalOffset *= -1
  icoPoints = icoPoints + [[math.cos(a)*planarDistance, verticalOffset, math.sin(a)*planarDistance] for a in angles]
  icoPoints.append([0,-1,0]) # Bottom point

  icoPoints = [np.asarray(p) for p in icoPoints] # Convert to numpy vectors

  ## Convert the points into an ommatidium
  # Calculate the acceptance angle for 1 steradian
  oneSteradianAcceptanceAngle = math.acos(-(1/(2*math.pi)-1)) * 2
  return [Ommatidium(np.zeros(3), p, oneSteradianAcceptanceAngle, 0.0) for p in icoPoints]

def _getEyeFeatures(line):
  data = [float(n) for n in line.split(" ")]
  position = np.asarray(data[:3])
  direction = np.asarray(data[3:6])
  acceptanceAngle = data[6]
  focalPointOffset = data[7]
  return (Ommatidium(position, direction, acceptanceAngle, focalPointOffset))
