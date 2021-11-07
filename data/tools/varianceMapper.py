DESCRIPTION = """A tool to map the variance of a given envrionment. It takes a scene, and runs a specially-designed insect eye through it, recording the sampling variance (or standard deviation) at each point on an N-by-N grid defined by a start position, grid-size and grid-step."""


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
  print("If you're running this from the */eye-renderer/data/tools folder, running 'export PYTHONPATH=\"$(cd ../../python-examples/ && pwd)\"' should fix it for you :)")
  exit()
import matplotlib.pyplot as plt

def getVariancesAtCurrentLocation(sampleCount, ommCount, renderer):
  renderer.renderFrame()
  frameData = renderer.getFramePointer()
  samples = np.copy(frameData[:,:,:3]) # Remove the alpha component
  avgImage = np.mean(samples, axis=0)
  differenceImages = samples - avgImage
  magnitudeImages = np.linalg.norm(differenceImages, axis=2)
  magnitudeSquaredImages = magnitudeImages * magnitudeImages
  varianceImage = np.sum(magnitudeSquaredImages, axis=0)/(sampleCount-1)
  return varianceImage


def main(argv):
  # Get the input parameters
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  parser.add_argument("-f, --file", type=str, required=True, metavar="FILE", dest="gltfPath", help="path to GlTF scene file.")
  parser.add_argument("-c, --center", type=float, metavar="N", default=[0,0,0], nargs=3, dest="centerLocation", help="The center of the analysis area, in cm.")
  parser.add_argument("-s, --scale", type=float, metavar="S", default=1.0, dest="mapScale", help="The scale, in cm, of the analysis square in the x/z plane (it's width/height).")
  parser.add_argument("-n, --step-number", type=int, metavar="N", default=10, dest="stepCount", help="The number of steps away from the center taken when sampling. It acts more like a square step-radius measure, e.g./ -n 10 results in a 21x21 grid of samples.")
  parser.add_argument("-m, --mode", type=str, metavar="MODE", default="max", choices=["max", "mean"], dest="mode", help="Configures the mode of variance analysis retrieval from a single view - accepts 'max' and 'mean', 'mean' averages all ommatidia's measures from the view, 'max' takes only the maximum. Defaults to 'max'")
  parser.add_argument("--standard-deviation", action="store_true", default=False, dest="calculateStandardDeviations")
  parser.add_argument("--visualise", action="store_true", default=False, dest="debugVis", help="Render to the renderer window a higher resolution compound-eye captured image from the position of the highest spread/visual frequency from each generation. Recommended used with a lower --spread-sample-count.")
  parser.add_argument("--spread-sample-count", type=int, metavar="SAMPLES", default=10000, dest="spreadSampleCount", help="The number of images taken from a given point using the same eye configuration in order to measure standard deviation across each ommatidium. Defaults to 10,000")
  parser.add_argument("--lib", type=str, metavar="PATH", nargs=1, default="", dest="libPath", help="Path to the eye render shared object (.so) file. Required if this python program has been moved. Checked before default relative paths to the make and ninja buildfolders.")
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

    # Store a copy of the original ommatidia for debug purposes
    originalOmmatidia = eyeTools.readEyeFile(eyeRenderer.getCurrentEyeDataPath())
    # Configure the compound eye to the standard 1sr icosahedron eye design
    uniformEyeData = eyeTools.getIcoOmmatidia()
    ommatidialCount = len(uniformEyeData)
    eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer, uniformEyeData)
    # Configure the eye to output the raw ommatidial display, and resize the output
    eyeRenderer.setCurrentEyeShaderName(c_char_p(b"raw_ommatidial_samples"))
    eyeTools.setRenderSize(eyeRenderer, ommatidialCount, parsedArgs.spreadSampleCount)
    eyeRenderer.setCurrentEyeSamplesPerOmmatidium(parsedArgs.spreadSampleCount)

    ## Move through each sample point and take a reading, storing it in the analysis map
    mapSize = parsedArgs.stepCount * 2 + 1
    totalSamples = mapSize**2
    stepScale = parsedArgs.mapScale/mapSize
    print("Step scale: ", stepScale)
    analysisMap = np.zeros((mapSize, mapSize))
    # Set the rotation to 0,0 and put it at the very beginning
    startPose = {"position":np.asarray(parsedArgs.centerLocation), "rotationAngles":np.zeros(3)}
    startOffset = parsedArgs.stepCount * stepScale * np.asarray([1,0,1])
    startPose["position"] = startPose["position"] - startOffset
    eyeRenderer.setCameraPose(*startPose["position"], *startPose["rotationAngles"])
    if not parsedArgs.debugVis:
      eyeRenderer.setVerbosity(False)
      print()
    startTime = time.time()
    for i,o in product(range(mapSize), range(mapSize)):
      # Move to the current location
      newLocation = startPose["position"] + np.asarray([i, 0, o]) * stepScale
      eyeRenderer.setCameraPose(*newLocation, *startPose["rotationAngles"])
      # Measure the variance there
      varianceImage = getVariancesAtCurrentLocation(parsedArgs.spreadSampleCount, ommatidialCount, eyeRenderer)
      if parsedArgs.mode == "mean":
        analysisMap[i,o] = np.mean(varianceImage)
      else:
        analysisMap[i,o] = np.max(varianceImage)

      # Debug view
      if parsedArgs.debugVis:
        # Configure to viewable image
        eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer, originalOmmatidia)
        eyeRenderer.setCurrentEyeShaderName(c_char_p(b"spherical_orientationwise"))
        eyeTools.setRenderSize(eyeRenderer,550,400)
        # Render
        eyeRenderer.renderFrame()
        eyeRenderer.displayFrame()
        time.sleep(0.5) # Brief pause
        # Reconfigure back to raw data using only 12 ommatidium
        eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer, uniformEyeData)
        eyeRenderer.setCurrentEyeShaderName(c_char_p(b"raw_ommatidial_samples"))
        eyeTools.setRenderSize(eyeRenderer, ommatidialCount, parsedArgs.spreadSampleCount)
      else:
        # Give a little loading bar
        sample = i*mapSize + o + 1
        percent = sample/totalSamples*100
        timeElapsed = time.time()-startTime
        timeLeft = timeElapsed/percent * 100 - timeElapsed
        timeLeftHours = math.floor(timeLeft/3600)
        timeLeftMins = math.floor(timeLeft/60) - timeLeftHours*60
        timeLeftSecs = timeLeft%60
        print("\rProcessed sample {} of {} ({:.2f}%, estimated time remaining: ~{:.0f}h {:.0f}m {:.0f}s)".format(sample, totalSamples, percent, timeLeftHours, timeLeftMins, timeLeftSecs), end='')
  
    if not parsedArgs.debugVis:
      print() # Print a new line to cancel out the time left display

    # Convert to standard deviations if it's set.
    if parsedArgs.calculateStandardDeviations:
      analysisMap = np.sqrt(analysisMap)

    print("Step scale: ", stepScale)
    # Now we have the map of all measures
    plt.imshow(analysisMap)
    plt.show()
    plt.imsave("scene-analysis.png", analysisMap)

    eyeRenderer.stop()
  except Exception as e:
    print(e)
    if eyeRenderer != None:
      eyeRenderer.stop()

if __name__ == "__main__":
  main(sys.argv[1:])
