# This script generates eye views and their position data and stores them in the data folder.
# Note that this can take a very long time to run, and it is recommended that you first try this with
# DEBUG set to true first so you can check what is being generated. The variables below
# DEBUG define things that can be changed including the eye names and variations (eyeTypes) of each to
# generate with. `framesToGenerate` specifies how many frames each eye variation should generate.


import matplotlib.pyplot as plt
import numpy as np
import time
import math
import itertools
from pathlib import Path
import os

# Insert the eye renderer # TODO: Put this as a submodule
import eyeRendererPaths
import sys

from ctypes import *
sys.path.insert(1, eyeRendererPaths.PYTHON_EXAMPLES_PATH)
import eyeRendererHelperFunctions as eyeTools

DEBUG = True

# Data paths
# Options: "AM_60185", "AM_60186", "BT_77966", "BT_77967", "BT_77970", "BT_77971", "BT_77973", "BT_77974":
eyeNames = ["AM_60186", "BT_77966", "BT_77967", "BT_77970", "BT_77971", "BT_77973", "BT_77974"]
# Options: "real", "single", "split"
eyeTypes = ["single", "split"]
framesToGenerate = 100000


dataTemplatePath = "eye-data/[EYE-NAME]-[EYE-TYPE].eye"
dataOutTemplatePath = "data/eye-distance-data/[EYE-NAME]/"
if DEBUG:
  dataOutTemplatePath = "data/quick-tests-data/[EYE-NAME]/"
  framesToGenerate = 10


# Set up the eye renderer
eyeRenderer = CDLL(eyeRendererPaths.EYE_RENDERER_LIB_PATH)
if DEBUG:
  print("Successfully loaded eye renderer: ", eyeRenderer)

eyeTools.configureFunctions(eyeRenderer)
eyeRenderer.loadGlTFscene(c_char_p(b"sim-environment/env_2.gltf"))
if DEBUG:
  eyeRenderer.gotoCameraByName(c_char_p(b"pano-cam"))
  #eyeRenderer.gotoCameraByName(c_char_p(b"compound-cam"))
  #eyeRenderer.setCurrentEyeSamplesPerOmmatidium(1000)
  eyeTools.setRenderSize(eyeRenderer, 550, 400)
else:
  eyeRenderer.gotoCameraByName(c_char_p(b"compound-cam"))
  eyeRenderer.setCurrentEyeShaderName(c_char_p(b"single_dimension_fast"))
  eyeTools.setRenderSize(eyeRenderer, 550, 400)
  #eyeTools.setRenderSize(eyeRenderer, OMMATIDIAL_COUNT, 1)
  eyeRenderer.setCurrentEyeSamplesPerOmmatidium(1000)

#for en in eyeNames for et in eyeTypes:
for en in eyeNames:
  for et in eyeTypes:
    # Configure the data path
    dataPath = dataTemplatePath.replace("[EYE-NAME]", en)
    dataPath = dataPath.replace("[EYE-TYPE]", et)
    dataPath = Path(dataPath)
    print("Loading", dataPath)

    dataPath.parent.mkdir(parents=True, exist_ok=True) # Make sure the path exists

    # Load the ommatidia
    eyeData = eyeTools.readEyeFile(dataPath) # Read in the eye data
    ommatidialCount = len(eyeData)
    print("\tOmmatidial count:", ommatidialCount)

    # Configure the output of the renderer
    eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer, eyeData)
    eyeRenderer.setCurrentEyeSamplesPerOmmatidium(3000)
    if DEBUG:
      eyeRenderer.displayFrame()
    if not DEBUG:
      eyeTools.setRenderSize(eyeRenderer, ommatidialCount, 1)

    # Start save files
    outPath = Path(dataOutTemplatePath.replace("[EYE-NAME]", en))
    outPath.mkdir(parents=True, exist_ok=True) # Make sure the path exists
    outPath /= et + "-distances.csv"

    # Make sure the folders for the view images are made
    (outPath.parent / et).mkdir(parents=True, exist_ok=True)

    fileMode = "w+"
    if outPath.is_file():
      print("\tFile already exists!")
      fileMode = "r+"
    with open(outPath, fileMode) as outF: # Open the output csv file to write to
      # First read through every line until we reach the end of the file so we know what index we're at
      currentIndex = 0
      while outF.readline().strip() != '':
        currentIndex += 1
      print(f"\tThere were already {currentIndex} entries for this eye.")

      for i in range(currentIndex, currentIndex + framesToGenerate):
        #Move to a random position
        relativePos = (np.random.random(3)*2-1) * (50/2)
        eyeRenderer.setCameraPosition(*relativePos) # Actually set the position
        # Render and save the frame
        renderTime = eyeRenderer.renderFrame() # Render the frame
        savepath = outPath.parent / et / f"{i}.ppm"
        eyeRenderer.saveFrameAs(c_char_p(str(savepath).encode("utf-8")) ) # Save the frame to the hard disk
        # Add the reference to the csv file
        outF.write("{:d},{:.20f},{:.20f},{:.20f}\n".format(i, *relativePos))

eyeRenderer.stop()
