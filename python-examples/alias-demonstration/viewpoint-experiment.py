import os.path
from pathlib import Path
from ctypes import *
from sys import platform
from numpy.ctypeslib import ndpointer
import numpy as np

from PIL import Image

try:
  import eyeRendererHelperFunctions as eyeTools
except Exception as e:
  print("Error importing eyeTools:", e)
  print("This is most likely because you do not have the 'python-examples' folder set as a path in $PYTHONPATH.")
  exit()

try:
  # Load the renderer
  eyeRenderer = CDLL("../../build/make/lib/libEyeRenderer3.so")
  print("Successfully loaded", eyeRenderer)

  # Configure the renderer's function outputs and inputs using the helper functions
  eyeTools.configureFunctions(eyeRenderer)

  #Load a scene
  print("Loading scene (please wait)...")
  eyeRenderer.loadGlTFscene(c_char_p(b"../../data/natural-standin-sky.gltf"))
  print("Scene loaded!")

  #Resize the renderer display in order to render the spherically-projected variable sample rate
  renderWidth = 700
  renderHeight = 300
  eyeTools.setRenderSize(eyeRenderer, renderWidth, renderHeight)

  # Create a numpy array to store the eye data
  aliasedToUnaliasedImage = np.zeros((renderHeight, renderWidth, 3), dtype=np.uint8)

  # Go to the 'insect-eye-spherical-projector' camera
  eyeRenderer.gotoCameraByName(c_char_p(b"insect-eye-spherical-projector"))
  #eyeRenderer.gotoCameraByName(c_char_p(b"insect-cam-1"))

  # Make sure the correct folders exist
  Path("output/generated-data/alias-demonstration/").mkdir(parents=True, exist_ok=True)
  Path("output/view-images/").mkdir(parents=True, exist_ok=True)

  # Iterate over eye sample counts
  for idx, s in enumerate(range(1,701)):
    samples = int(s/1)
    eyeRenderer.setCurrentEyeSamplesPerOmmatidium(samples)
    eyeRenderer.renderFrame() # First call to ensure randoms are configured
    renderTime = eyeRenderer.renderFrame() # Second call to actually render the image
    #print("Rendered with {:n} in {d:} milliseconds.".format(s, renderTime))

    #Save the frame as a .ppm file into a backup storage location (incase we need to use them or something)
    eyeRenderer.saveFrameAs(c_char_p(("output/generated-data/alias-demonstration/spherical-image-"+str(samples)+"-samples.ppm").encode()))

    # Retrieve the data
    frameData = eyeRenderer.getFramePointer()
    frameDataRGB = frameData[:,:,:3] # Remove the alpha component
    frameDataRGB = np.flipud(frameDataRGB) # Flip the data the right way up
    aliasedToUnaliasedImage[:,idx,:] = np.copy(frameDataRGB[:, idx, :])

  combinedImg = Image.fromarray(aliasedToUnaliasedImage, "RGB")
  combinedImg.show()
  combinedImg.save("output/view-images/combinedImage-700segs.png")

  eyeRenderer.stop() # Stop the renderer

except Exception as e:
  print(e)
