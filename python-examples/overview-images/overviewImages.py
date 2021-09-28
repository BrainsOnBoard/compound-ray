import math
import sys
from itertools import product
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
  print("If you're running this from the */eye-renderer/*/* folder, running 'export PYTHONPATH=\"$(cd ../../python-examples/ && pwd)\"' should fix it for you :)")
  exit()
import matplotlib.pyplot as plt

def makeWeirdEye():
  rows = 20
  cols = 30
  height = 1.5
  radius = 0.8
  arc = math.pi/1.2
  totalSlant = 0.2
  curl = 0.3
  curlSeverity = 2
  topSideScale = 0.85
  jiggleAngle = (math.pi/4)/(cols*2)
  omms = []
  for r,c in product(range(rows), range(cols)):
    angle = -arc + (arc/(cols+1))*c + (r%2)*jiggleAngle
    xProj = math.sin(angle)
    zProj = math.cos(angle)
    scaledRadius = radius * (1 - ((r/rows) * (1-topSideScale)))
    position = np.asarray([radius*xProj, (height/(rows+1))*r, radius*zProj])
    position = position + np.asarray([0,1,0])*(height/(2*(rows+1)))*(c%2)
    direction = np.asarray([xProj, 0, zProj])
    acceptanceAngle = 2/180*math.pi
    focalpointOffset = 0
    omm = eyeTools.Ommatidium(position,direction,acceptanceAngle,focalpointOffset)
    if omm.position[1] > height-curl:
      omm.direction[1] = -((omm.position[1]-(height-curl))/curl)*curlSeverity
      omm.direction = omm.direction/np.linalg.norm(omm.direction)
    omms.append(omm)
  for omm in omms:
    omm.position[0] += (omm.position[1]/height)*totalSlant
  return omms

def main(argv):
  # Get the input parameters
  parser = argparse.ArgumentParser()
  parser.add_argument("-f, --file", type=str, required=True, metavar="FILE", dest="gltfPath", help="path to GlTF scene file.")
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

    # Get/make the ommatidial layouts
    uniformOmms = eyeTools.readEyeFile("../../data/eyes/1000-equidistant.eye")
    acuteOmms = eyeTools.readEyeFile("../../data/eyes/1000-horizontallyAcute-variableDegree.eye")
    weirdOmms = makeWeirdEye()
    # Quick save the weird eye
    print("ommlen",len(weirdOmms))
    eyeTools.saveEyeFile("weirdEye.eye", weirdOmms)

    eyeRenderer.setCurrentEyeSamplesPerOmmatidium(600)

    # Render from the different eyes
    eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer, uniformOmms)
    eyeRenderer.renderFrame()
    eyeRenderer.displayFrame()
    eyeRenderer.saveFrameAs(c_char_p(("uniform-omms.ppm").encode()))

    eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer, acuteOmms)
    eyeRenderer.renderFrame()
    eyeRenderer.displayFrame()
    eyeRenderer.saveFrameAs(c_char_p(("acute-omms.ppm").encode()))

    eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer, weirdOmms)
    eyeRenderer.renderFrame()
    eyeRenderer.displayFrame()
    eyeRenderer.saveFrameAs(c_char_p(("weird-omms.ppm").encode()))

    eyeRenderer.stop()
  except Exception as e:
    print(e)
    if eyeRenderer != None:
      eyeRenderer.stop()

if __name__ == "__main__":
  main(sys.argv[1:])
