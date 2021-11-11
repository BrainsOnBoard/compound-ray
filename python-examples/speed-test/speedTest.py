import math
import sys
from itertools import product
import argparse
import pathlib
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import GPUtil
from ctypes import *
try:
  import eyeRendererHelperFunctions as eyeTools
except Exception as e:
  print("Error importing eyeTools:", e)
  print("This is most likely because you do not have the 'python-examples' folder set as a path in $PYTHONPATH.")
  print("If you're running this from the */compound-ray/data/tools folder, running 'export PYTHONPATH=\"$(cd ../../python-examples/ && pwd)\"' should fix it for you :)")
  exit()

def main(argv):
  # Get the input parameters
  parser = argparse.ArgumentParser()
  parser.add_argument("-f, --file", type=str, required=True, metavar="FILE", dest="gltfPath", help="path to GlTF scene file.")
  parser.add_argument("--lib", type=str, metavar="PATH", default="", dest="libPath", help="Path to the eye render shared object (.so) file. Required if this python program has been moved. Checked before default relative paths to the make and ninja buildfolders.")
  parser.add_argument("-o, --output-filename", type=str, metavar="OUTFILE", default="output", dest="outfileName", help="The filename for output files (exclude file extensions, defaults to 'output')")
  parser.add_argument("-s, --samples", type=int, metavar="N", default=100, dest="sampleCount", help="The number of images to take in order to calculate the average render time.")
  parser.add_argument("-m, --max-sample-rays", type=int, metavar="N", default=100, dest="maxSampleRays", help="The maximum number of per-ommatidium samples to measure render speed until.")
  parser.add_argument("-n, --min-samples-rays", type=int, metavar="N", default=1, dest="minSampleRays", help="The minimum number of per-ommatidium samples to measure render speed until.")
  parser.add_argument("-v, --verbose", action="store_true", default=False, dest="printGraphs", help="Be verbose. Print the graphs at the end of a run (halting).")
  parser.add_argument("-g, --GPU-name", type=str, metavar="NAME", default="", dest="GPUnameOverride", help="Used to override the GPU name if GPUtil is failing.")
  parsedArgs = parser.parse_args()

  eyeRenderer = None

  # Get GPU name
  gpuName = parsedArgs.GPUnameOverride
  if gpuName == "":
    gpuName = GPUtil.getGPUs()[0].name.replace(" ","_")


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


    # Find a compound eye in the scene, go to it.
    eyeTools.gotoFirstCompoundEye(eyeRenderer)
    print("Compound eye found.")

    # Load in a known eye design with equal sampling in all directions
    eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer, eyeTools.readEyeFile("../scenes/1000-equidistant.eye"))
    print("Eye ommatidia set.")

    # Configure the eye to be a single dimensional output
    eyeRenderer.setCurrentEyeShaderName(c_char_p(b"single_dimension_fast"))
    eyeTools.setRenderSize(eyeRenderer, eyeRenderer.getCurrentEyeOmmatidialCount(), 1)
    print("Eye configured.")
    
    # Render frames for 10 seconds at the start to account for (presumably) GAS caching
    # that sets in after ~0.5 seconds. As this is a measure of the average per-frame
    # render time, this will be ignored.
    print("Performing 10-second frame purge...")
    eyeRenderer.setVerbosity(False)
    eyeRenderer.setCurrentEyeSamplesPerOmmatidium(1)
    timeAtStart = time.time()
    while time.time()-timeAtStart <= 10:
      eyeRenderer.renderFrame()

    # Iterate over the range of samples, measuring average computation times
    print("Beginning speed test for {} ({} to {} rays, {} samples per rayset)...".format(parsedArgs.outfileName, parsedArgs.minSampleRays, parsedArgs.maxSampleRays, parsedArgs.sampleCount))
    avgSpeeds = np.zeros(parsedArgs.maxSampleRays)
    upperBound = parsedArgs.maxSampleRays+1
    rayCounts = np.arange(parsedArgs.minSampleRays,upperBound)
    # Estimated time remaining calculation storage
    startTime = time.time()
    allTimeRaysSent = 0
    timeRemainingSecs = 0
    for rayCount in rayCounts:
      # Set the per-ommatidial ray count
      eyeRenderer.setCurrentEyeSamplesPerOmmatidium(rayCount)
      # First render the initial ray to generate new randoms
      eyeRenderer.renderFrame()
      eyeRenderer.renderFrame()
      totalRenderingTime = 0
      for s in range(parsedArgs.sampleCount):
        totalRenderingTime += eyeRenderer.renderFrame() # Accumulate the total rendering time
        if s%30 or s==parsedArgs.sampleCount-1:
          # Update the readout, but only occassionally
          timeRemainingSeconds = timeRemainingSecs%60
          timeRemainingMins = (timeRemainingSecs/60)%60
          timeRemainingHours = timeRemainingSecs/3600
          print("\rGPU: {}\tCurrent ray count: {} ({} out of {}, {:.2f}%, ~{:.0f}h {:.0f}m {:.0f}s remaining)".format(gpuName, rayCount, rayCount-parsedArgs.minSampleRays+1, upperBound-parsedArgs.minSampleRays, ((rayCount-parsedArgs.minSampleRays)*parsedArgs.sampleCount+s)/((upperBound-parsedArgs.minSampleRays)*parsedArgs.sampleCount)*100, timeRemainingHours, timeRemainingMins, timeRemainingSeconds), end='')
      # store the average per-frame rendering time
      avgSpeeds[rayCount-1] = totalRenderingTime/parsedArgs.sampleCount
      # Add up some running data for estimated time remaining calculation (Note time estimation isn't going to be accurate b/c of randoms allocation time)
      allTimeRenderingTime = time.time()-startTime
      allTimeRaysSent += parsedArgs.sampleCount*rayCount
      timeRemainingSecs = (allTimeRenderingTime/allTimeRaysSent) * (upperBound-rayCount) * 0.5*(rayCount + parsedArgs.maxSampleRays) * parsedArgs.sampleCount
    print()
    eyeRenderer.stop()

    np.savetxt("{}-{}-frame-rendertime-averages-({}-{}-rays,{}-samples).txt".format(gpuName,parsedArgs.outfileName,parsedArgs.minSampleRays, parsedArgs.maxSampleRays,parsedArgs.sampleCount), avgSpeeds, delimiter=",")
    avgFPSs = 1000/avgSpeeds
    np.savetxt("{}-{}-frame-rendertime-average-FPSs-({}-{}-rays,{}-samples).txt".format(gpuName,parsedArgs.outfileName,parsedArgs.minSampleRays,parsedArgs.maxSampleRays,parsedArgs.sampleCount), avgFPSs, delimiter=",")
    print("Results saved.")

    if parsedArgs.printGraphs:
      plt.plot(rayCounts, avgSpeeds)
      plt.show()
      plt.xscale("log")
      plt.plot(rayCounts, avgSpeeds)
      plt.show()

      plt.plot(rayCounts, avgFPSs)
      plt.show()
      plt.xscale("log")
      plt.plot(rayCounts, avgFPSs)
      plt.show()

  except Exception as e:
    print(e)
    if eyeRenderer != None:
      eyeRenderer.stop()

if __name__ == "__main__":
  main(sys.argv[1:])
