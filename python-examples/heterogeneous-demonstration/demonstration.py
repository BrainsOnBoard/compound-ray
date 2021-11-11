DESCRIPTION="""Renders a given environment through a given .eye file,
first through it as it is, then through it but with all acceptance angles set to the minimum, then all
set to the max found in the file."""

import math
import sys
import os.path
from itertools import product
from functools import reduce
import argparse
import pathlib
import numpy as np
import time
from ctypes import *
try:
  import eyeRendererHelperFunctions as eyeTools
except Exception as e:
  print("Error importing eyeTools:", e)
  print("This is most likely because you do not have the 'python-examples' folder set as a path in $PYTHONPATH.")
  print("If you're running this from the */compound-ray/data/tools folder, running 'export PYTHONPATH=\"$(cd ../../python-examples/ && pwd)\"' should fix it for you :)")
  exit()
import matplotlib.pyplot as plt

def main(argv):
  # Get the input parameters
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  parser.add_argument("-f, --file", type=str, required=True, metavar="FILE", dest="gltfPath", help="Path to GlTF scene file.")
  parser.add_argument("-e, --eye", type=str, metavar="FILE", dest="eyePath", default="1000-extreme-horizontallyAcute-variableDegree.eye", help="Specifies a path to an eye file to load into the scene.")
  parser.add_argument("--lib", type=str, metavar="PATH", default="", dest="libPath", help="Path to the eye render shared object (.so) file. Required if this python program has been moved. Checked before default relative paths to the make and ninja buildfolders.")
  parsedArgs = parser.parse_args()

  eyeRenderer = None

  try:
    # First find the eye lib in a slightly more robust manner
    eyeLibPath = parsedArgs.libPath
    if eyeLibPath == "" or not pathlib.Path(eyeLibPath).exists():
      # If the user didn't specify a lib path, then search the two potential locations
      scriptDir = pathlib.Path(__file__).parent.absolute()
      makePath  = scriptDir/pathlib.Path("../../build/make/lib/libEyeRenderer3.so")
      ninjaPath = scriptDir/pathlib.Path("../../build/ninja/lib/libEyeRenderer3.so")
      if makePath.exists():
        eyeLibPath = makePath
      elif ninjaPath.exists():
        eyeLibPath = ninjaPath
      else:
        print("ERROR: None of the supplied paths seemed to contain libEyeRenderer3.so. Paths:")
        print("user supplied   :", parsedArgs.libPath)
        print("make build path :", makePath)
        print("ninja build path:", ninjaPath)

    # Load the renderer
    eyeRenderer = CDLL(eyeLibPath)
    print("Loaded eye renderer lib:", eyeRenderer)

    # Configure the renderer's function outputs and inputs using the helper functions
    eyeTools.configureFunctions(eyeRenderer)
    
    # Load the GlTF scene:
    print("Loading scene at {} (please wait)...".format(parsedArgs.gltfPath))
    # Check the file exists
    gltfPath = pathlib.Path(parsedArgs.gltfPath)
    if not gltfPath.exists():
      raise Exception("Error: Supplied gltf file does not exist at " + str(gltfPath))
    eyeRenderer.loadGlTFscene(c_char_p(str(gltfPath).encode("utf-8")))
    print("Scene loaded!")

    eyeTools.setRenderSize(eyeRenderer, 550, 400)

    # Find a compound eye in the scene, go to it.
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

    # Get the heterogeneous ommatidial layout (from local file)
    heterogeneousLayout = eyeTools.readEyeFile(parsedArgs.eyePath)

    # Make the big acceptance angle copy
    bigGlobalAcceptanceAngle = [omm.copy() for omm in heterogeneousLayout]
    biggestAcceptanceAngle = reduce((lambda biggest, omm: omm.acceptanceAngle if omm.acceptanceAngle > biggest else biggest), bigGlobalAcceptanceAngle, 0.0)
    for omm in bigGlobalAcceptanceAngle:
      omm.acceptanceAngle = biggestAcceptanceAngle
    print("Largest acceptance angle found in eye file:", biggestAcceptanceAngle)

    # Make the small acceptance angle copy
    smallGlobalAcceptanceAngle = [omm.copy() for omm in heterogeneousLayout]
    smallestAcceptanceAngle = reduce((lambda smallest, omm: omm.acceptanceAngle if omm.acceptanceAngle < smallest else smallest), smallGlobalAcceptanceAngle, float("inf"))
    for omm in smallGlobalAcceptanceAngle:
      omm.acceptanceAngle = smallestAcceptanceAngle
    print("Smallest acceptance angle found in eye file:", smallestAcceptanceAngle)

    # Increase samples so it's not all janky
    eyeRenderer.setCurrentEyeSamplesPerOmmatidium(1000)

    # Render from the three different eye layouts

    # Work out how many times this has been run so you don't duplicate or override output files
    runIndex = 0
    while os.path.isfile("heterogeneous-omms-{}.ppm".format(runIndex)):
      runIndex = runIndex+1

    eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer, heterogeneousLayout)
    eyeRenderer.renderFrame()
    eyeRenderer.displayFrame()
    eyeRenderer.saveFrameAs(c_char_p(("heterogeneous-omms-{}.ppm".format(runIndex)).encode()))

    eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer, bigGlobalAcceptanceAngle)
    eyeRenderer.renderFrame()
    eyeRenderer.displayFrame()
    eyeRenderer.saveFrameAs(c_char_p(("homogeneous-omms-big-{}.ppm".format(runIndex)).encode()))

    eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer, smallGlobalAcceptanceAngle)
    eyeRenderer.renderFrame()
    eyeRenderer.displayFrame()
    eyeRenderer.saveFrameAs(c_char_p(("homogeneous-omms-small-{}.ppm".format(runIndex)).encode()))

    eyeRenderer.stop()
  except Exception as e:
    print(e)
    if eyeRenderer != None:
      eyeRenderer.stop()

if __name__ == "__main__":
  main(sys.argv[1:])
