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

# Updates the render output size while updating the return type of the render pointer
def setRenderSize(eyeRenderer, width, height):
  eyeRenderer.setRenderSize(width, height)
  eyeRenderer.getFramePointer.restype = ndpointer(dtype=c_ubyte, shape = (width, height, 4))

# Sets the samples per ommatidium for the current compound eye in a safer manner,
# i.e. It checks if this is a compound eye, and re-renders to recalculate random seeds
def setSamplesPerOmmatidium(eyeRenderer, samples):
  if(eyeRenderer.isCompoundEyeActive()):
    eyeRenderer.setCurrentEyeSamplesPerOmmatidium(samples)
    eyeRenderer.renderFrame()
