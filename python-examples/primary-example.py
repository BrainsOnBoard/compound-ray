import os.path
import time
from ctypes import *
from sys import platform
from numpy.ctypeslib import ndpointer
import numpy as np

from PIL import Image

import eyeRendererHelperFunctions as eyeTools

# Makes sure we have a "test-images" folder
if not os.path.exists("test-images"):
    os.mkdir("test-images")

sleepTime = 5 # How long to sleep between rendering images

try:
  # Load the renderer
  eyeRenderer = CDLL("../build/make/lib/libEyeRenderer3.so")
  print("Successfully loaded ", eyeRenderer)

  # Configure the renderer's function outputs and inputs using the helper functions
  eyeTools.configureFunctions(eyeRenderer)

  # Load a scene
  eyeRenderer.loadGlTFscene(c_char_p(b"../data/ofstad-arena/ofstad-acceptance-angle.gltf"))

  # Resize the renderer display
  # This can be done at any time, but restype of getFramePointer must also be updated to match as such:
  renderWidth = 200
  renderHeight = 200
  eyeRenderer.setRenderSize(renderWidth,renderHeight)
  eyeRenderer.getFramePointer.restype = ndpointer(dtype=c_ubyte, shape = (renderHeight, renderWidth, 4))
  # An alternative to the above two lines would be to run:
  #eyeTools.setRenderSize(eyeRenderer, renderWidth, renderHeight)

  # Iterate through a few cameras and do some stuff with them
  for i in range(5):
    # Actually render the frame
    renderTime = eyeRenderer.renderFrame()
    print("View from camera '", eyeRenderer.getCurrentCameraName(), " rendered in ", renderTime)
    
    eyeRenderer.displayFrame() # Display the frame in the renderer

    # Save the frame as a .ppm file directly from the renderer
    eyeRenderer.saveFrameAs(c_char_p(("test-images/test-image-"+str(i)+".ppm").encode()))

    # Retrieve frame data
    # Note: This data is not owned by Python, and is subject to change
    # with subsequent calls to the renderer so must be deep-copied if
    # you wish for it to persist.
    frameData = eyeRenderer.getFramePointer()
    frameDataRGB = frameData[:,:,:3] # Remove the alpha component
    print("FrameData type:", type(frameData))
    print("FrameData:\n",frameData)
    print("FrameDataRGB:\n",frameDataRGB)

    # Use PIL to display the image (note that it is vertically inverted)
    img = Image.fromarray(frameDataRGB, "RGB")
    img.show()

    # Vertically un-invert the array and then display
    rightWayUp = np.flipud(frameDataRGB)
    #rightWayUp = frameDataRGB[::-1,:,:] also works
    img = Image.fromarray(rightWayUp, "RGB")
    img.show()

    # If the current eye is a compound eye, set the sample rate for it high and take another photo
    if(eyeRenderer.isCompoundEyeActive()):
      print("This one's a compound eye, let's get a higher sample rate image!")
      eyeRenderer.setCurrentEyeSamplesPerOmmatidium(100);
      renderTime = eyeRenderer.renderFrame() # Render the frame
      eyeRenderer.saveFrameAs(c_char_p(("test-images/test-image-"+str(i)+"-100samples.ppm").encode()))# Save it
      Image.fromarray(eyeRenderer.getFramePointer()[::-1,:,:3], "RGB").show() # Show it in PIL (the right way up)

      ## Change this compound eye's ommatidia to only be the first 10 in the list:
      #time.sleep(5)

      #ommList = eyeTools.readEyeFile(eyeRenderer.getCurrentEyeDataPath())
      #eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer,ommList[:10])

      #eyeRenderer.renderFrame()
      #eyeRenderer.displayFrame()

      ## Put it back
      #eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer,ommList)
      #eyeRenderer.renderFrame()
      #eyeRenderer.displayFrame()

    print("Sleeping for " + str(sleepTime) + " seconds...")

    # Change to the next Camera
    eyeRenderer.nextCamera()
    time.sleep(sleepTime)

  # Finally, stop the eye renderer
  eyeRenderer.stop()
except Exception as e:
  print(e);


