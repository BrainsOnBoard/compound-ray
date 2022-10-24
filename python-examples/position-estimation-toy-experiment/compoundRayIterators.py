###################################################################
#
#  An iterator that iterates over a cube sampling at uniform points
#
###################################################################
#
# Not designed to be run on it's own (although it can be) - this file
# contains classes that can be used (as exampled in main below) to
# iterate over a sampling cube to generate samples that are either
# randomly or uniformly distributed.
#
# Ensure that you have set PYTHON_EXAMPLES_PATH and EYE_RENDERER_LIB_PATH
# in eyeRendererPaths before running

import numpy as np
import time
import torch
import math

# Insert the eye renderer
import eyeRendererPaths
import sys
from ctypes import *
sys.path.insert(1, eyeRendererPaths.PYTHON_EXAMPLES_PATH)
import eyeRendererHelperFunctions as eyeTools

class CompoundRayIterator:
  def __init__(self, eyeFilepath, debug=False, debugPano=True, transform=None, resultNormalisationData=None):
    self.debug = debug

    # Load the eye renderer library
    self.eyeRenderer = CDLL(eyeRendererPaths.EYE_RENDERER_LIB_PATH)
    if debug:
      print("Successfully loaded eye renderer: ", self.eyeRenderer)


    # Load the scene
    eyeTools.configureFunctions(self.eyeRenderer)
    self.eyeRenderer.loadGlTFscene(c_char_p(b"sim-environment/env_2.gltf"))

    if (debug == False):
      self.eyeRenderer.setVerbosity(False)

    # Load in the compound eye configuration
    self.eyeRenderer.gotoCameraByName(c_char_p(b"compound-cam")) # Go to the camera
    eyeConfig = eyeTools.readEyeFile(eyeFilepath) # Load the configuration
    print(f"EYE RENDERER LEN {len(eyeConfig)}")
    eyeTools.setOmmatidiaFromOmmatidiumList(self.eyeRenderer,eyeConfig) # Set the configuration
    self.eyeRenderer.setCurrentEyeShaderName(c_char_p(b"single_dimension_fast")) # Set the ouput shader
    eyeTools.setRenderSize(self.eyeRenderer, len(eyeConfig), 1) # Set the eye renderer output size
    self.eyeRenderer.setCurrentEyeSamplesPerOmmatidium(1000) # Set the per-ommatidial samples to some arbitrarily large number

    # Configure camera selection
    if debug:
      # If in debug mode optionally switch into panoramic or compound re-projection modes
      if debugPano:
        self.eyeRenderer.gotoCameraByName(c_char_p(b"pano-cam"))
      else:
        self.eyeRenderer.setCurrentEyeShaderName(c_char_p(b"spherical_split_orientationwise"))
      eyeTools.setRenderSize(self.eyeRenderer, 550, 400)

    # Store the result transform data
    if(resultNormalisationData != None):
      self.resultNormalisationData = resultNormalisationData
    else:
      self.resultNormalisationData = None

    # Store the data transform data
    if(transform != None):
      self.tf = transform
    else:
      self.tf = None

  # Destructor
  def __del__(self):
    # Make sure the eye renderer shuts down properly when this object is destroyed
    self.eyeRenderer.stop()

  # Iteration start/configuration
  def __iter__(self):
    return(self)

# Each output is the camera in a random location within the sampling cube
class RandomCubeIterator(CompoundRayIterator):
  def __init__(self, eyeFilepath, debug=False, cubeSize=50, debugPano=True, transform=None, resultNormalisationData=None):
    super().__init__(eyeFilepath, debug, debugPano, transform=transform, resultNormalisationData=resultNormalisationData)
    self.cubeSize = cubeSize

  # Get item function
  def __next__(self):
    # Put the camera in a random position within a 50mm^3 box around the point object
    relativePos = (np.random.random(3)*2-1) * (self.cubeSize/2)
    self.eyeRenderer.setCameraPosition(*relativePos) # Actually set the position

    renderTime = self.eyeRenderer.renderFrame() # Render the frame

    # Get the frame data
    if self.debug:
      self.eyeRenderer.displayFrame()
      time.sleep(0.1)
    image = np.copy(self.eyeRenderer.getFramePointer()[:,:,:3]) # Copy out the image (minus the alpha channel)
    return torch.from_numpy(image.astype(np.dtype("f"))), torch.from_numpy(relativePos.astype(np.dtype("f")))

# Each output is the camera at a uniform sampling location within the sampling cube
class UniformCubeIterator(CompoundRayIterator):
  def __init__(self, eyeFilepath, debug=False, cubeSize=50, samplingSize=100, debugPano=True, transform=None, resultNormalisationData=None):
    super().__init__(eyeFilepath, debug, debugPano, transform=transform, resultNormalisationData=resultNormalisationData)
    self.cubeSize = cubeSize
    self.samplingSize = samplingSize

  # Override the iter function to add a tracker for the current positions
  def __iter__(self):
    self.sampleID = 0
    self.sampleGap = self.cubeSize/(self.samplingSize+1) # Get the distance between each sample point
    allSamplesWidth = self.samplingSize*self.sampleGap
    self.startPos = np.ones(3)*(-allSamplesWidth/2)
    return(self)

  # Get item function
  def __next__(self):
    # Convert the sample ID to a 3D position
    zPos = math.floor(self.sampleID/(self.samplingSize**2))
    yPos = math.floor((self.sampleID - zPos*(self.samplingSize**2)) / self.samplingSize)
    xPos = (self.sampleID - zPos*(self.samplingSize**2) - yPos*self.samplingSize)
    coord = np.asarray([xPos, yPos, zPos], dtype=np.int32)

    # Use coord to calculate the position of the sample point to the center
    samplingPos = coord*np.ones(3)*self.sampleGap + self.startPos
    self.eyeRenderer.setCameraPosition(*samplingPos) # Actually set the position

    # Increment the sample ID
    self.sampleID = (self.sampleID + 1)%(self.samplingSize**3)

    renderTime = self.eyeRenderer.renderFrame() # Render the frame

    # Get the frame data
    if self.debug:
      print("Rendering at: {}, {}, {}".format(*coord))
      self.eyeRenderer.displayFrame()
      time.sleep(0.1)
    #image = np.copy(self.eyeRenderer.getFramePointer()[:,:,:3]) # Copy out the image (minus the alpha channel)
    image = np.copy(self.eyeRenderer.getFramePointer()[:,:,0]) # Copy out the image (minus the alpha channel)

    imageOut = torch.from_numpy(image.astype(np.dtype("f")))
    vectorOut = torch.from_numpy(samplingPos.astype(np.dtype("f")))

    if(self.tf != None):
      imageOut = self.tf(image.astype(np.dtype("f"))) # Here we actually overwrite the previous "from_numpy" because this only wants numpy or PIL images for some reason
    if(self.resultNormalisationData != None):
      vectorOut = (vectorOut - self.resultNormalisationData["means"])/self.resultNormalisationData["stds"]
    return imageOut, vectorOut, coord

  def getSamplingSize(self):
    return self.samplingSize

  def getTotalSamplePoints(self):
    return self.samplingSize**3

if __name__ == "__main__":
  print("Random iterator:")
  randomIterator = iter(RandomCubeIterator("sim-environment/eyes/AM_60185-real.eye", debug=False, debugPano=False))
  start = time.time()
  count = 100000
  for i in range(count):
    next(randomIterator)
    #time.sleep(10)
  print("Finished {} renders in {} seconds".format(count, time.time() - start))
  #print("Uniform iterator:")
  #uniformIter = UniformCubeIterator("sim-environment/eyes/AM_60185-real.eye", debug=True, debugPano=False, samplingSize=4, cubeSize = 10)
  #uniformIterator = iter(uniformIter)
  #for i in range(4**3):
  #  print("THE THING:")
  #  data = next(uniformIterator)
  #  print(data)
  #  #a,b = data
  #  a = data[0]
  #  b = data[1]
  #  print("A:", a)
  #  print("B:", b)
  #  time.sleep(1)
  #print("Uniform iterator 2:")
  #uniformIterator = iter(uniformIter)
  #for i in range(4**3):
  #  next(uniformIterator)
