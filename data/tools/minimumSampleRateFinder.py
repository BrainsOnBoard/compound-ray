DESCRIPTION = """A tool to find the minimum sample rate of a given gltf scene and minimum per-frame percentage change, in samples per steradian.
It does this by searching the given gltf scene (within configured bounds) for the point of highest visual frequency within a scene,
and then increasing the per-ommatidial sampling rate until frame-to-frame differenes are, on average, no more than N% (difference in standard deviation).
The output of this can be used to calculate the minimum sample number you should expect to use with a given .eye file, when used with the eyeSampleRate tool.
"""

import time
import argparse
import sys
import numpy as np
import pathlib
import math
from ctypes import *
try:
  import eyeRendererHelperFunctions as eyeTools
except Exception as e:
  print("Error importing eyeTools:", e)
  print("This is most likely because you do not have the 'python-examples' folder set as a path in $PYTHONPATH.")
  print("If you're running this from the */compound-ray/data/tools folder, running 'export PYTHONPATH=\"$(cd ../../python-examples/ && pwd)\"' should fix it for you :)")
  exit()

from PIL import Image
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

def getVarianceAtCurrentLocationWithSpecifiedSamplesPerOmm(renderer, samplesPerOmm, sampleCount):
  renderer.setCurrentEyeSamplesPerOmmatidium(samplesPerOmm)
  allImages = np.copy(renderer.getFramePointer()[:,:,:3])
  for i in range(sampleCount-1):
    renderer.renderFrame()
    allImages = np.vstack((allImages, np.copy(renderer.getFramePointer()[:,:,:3])))
  avgImage = np.mean(allImages, axis=0)
  differenceImages = allImages - avgImage
  magnitudeImages = np.linalg.norm(differenceImages, axis=2)
  magnitudeSquaredImages = magnitudeImages * magnitudeImages
  varianceImage = np.sum(magnitudeSquaredImages, axis=0)/(sampleCount-1)
  return varianceImage

positioningTimeTotal = 0
renderingTimeTotal = 0

def boundlessBounding(pose):
  return pose
def boxBounding(pose):
  
  pose["position"] = np.maximum(pose, )

def main(argv):
  # Get the input parameters
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  parser.add_argument("-f, --file", type=str, required=True, metavar="FILE", dest="gltfPath", help="path to GlTF scene file.")
  parser.add_argument("-p, --percent", type=float, metavar="PERCENT", default=0.01, dest="cutoffPercent", help="The cuttoff percentage the program will increase sample count until the maximum standard deviation is within, relative to the maximum difference (of the distance from [0,0,0] to [255,255,255]). 0.05 (5 percent) by default.")
  parser.add_argument("-s, --search-bound", type=str, metavar="BOUND TYPE", default="boundless", choices=["box", "cylinder", "boundless"], dest="boundType", help="Configures the type of search bound to use. Options are 'box', 'cylinder' or 'boundless'")
  parser.add_argument("-c, --search-cylinder", type=float, default=[0,0,0,0,0], metavar="N", nargs=5, dest="searchCylinder", help="The specifications of a search bounding sphere - defined in the form of a single X,Y,Z coordinate for the center of the base of the cylinder and a following radius and height of the cylinder, e.g./ -c 0 0 0 2 5")
  parser.add_argument("-b, --search-box", type=float, default=[0,0,0,0,0,0], metavar="N", nargs=6, dest="searchBox", help="The coordinates - in the form of two sets of X,Y,Z coordinates (lowest, then highest), of a search bounding box, e.g./ -b 0 0 0 2 3 3")
  parser.add_argument("--lib", type=str, metavar="PATH", default="", dest="libPath", help="Path to the eye render shared object (.so) file. Required if this python program has been moved. Checked before default relative paths to the make and ninja buildfolders.")
  parser.add_argument("--spread-sample-count", type=int, metavar="SAMPLES", default=1000, dest="spreadSampleCount", help="The number of images taken from a given point using the same eye configuration in order to measure standard deviation across each ommatidium.")
  parser.add_argument("--generation-size", type=int, metavar="S", default=10000, dest="GAgenerationSize", help="Generation size of the genetic algorithm for finding the point of highest spread/visual frequency.")
  parser.add_argument("--visualise", action="store_true", default=False, dest="debugVis", help="Render to the renderer window a higher resolution compound-eye captured image from the position of the highest spread/visual frequency from each generation.")
  parser.add_argument("--genetic-search-cutoff", type=float, metavar="P", dest="searchCutoff", default=0.001, help="The point at which the genetic search will be halted if the current highest variance is within P times the previous highest. Defaults to 0.001 (0.1 percent).")
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

    # Set the render size
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

    originalOmmatidia = eyeTools.readEyeFile(eyeRenderer.getCurrentEyeDataPath())
    print("Backed up original ommatidia.")

    ### Find the point of highest visual frequency
    ## Set the compound eye to a special design that's got a set number of equidistantly-spaced ommatidia.
    uniformEyeData = eyeTools.getIcoOmmatidia()
    ommatidialCount = len(uniformEyeData)
    eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer, uniformEyeData)
    ## Configure the eye to output the raw ommatidial display, and resize the output
    eyeRenderer.setCurrentEyeShaderName(c_char_p(b"raw_ommatidial_samples"))
    eyeTools.setRenderSize(eyeRenderer,ommatidialCount,parsedArgs.spreadSampleCount)
    eyeRenderer.setCurrentEyeSamplesPerOmmatidium(parsedArgs.spreadSampleCount)

    ## Use simple GA (translation, small axis-angle rotation [random axis, maximum angle to angle between two points on the isosphere]) to search the space for point of max spread (highest visual freq.)
    ##   Carry on going until the change in max variance from the previous one is below M% (Note: Different to the similar metric to use for the next step)
    def rnd3vec(scale):
      """Returns a random 3vector with values between -scale and +scale"""
      return np.random.uniform(-1,1,size=3)*scale
    global positioningTimeTotal
    global renderingTimeTotal
    def getMaxVarianceAtPose(pose):
      """Moves a the camera to the pose, then calculates the maximum variance at that location from all variances at all eyes"""
      # First reset the camera, and then move it to the pose
      global positioningTimeTotal
      global renderingTimeTotal
      timeBefore = time.time()
      eyeRenderer.setCameraPose(*pose["position"], *pose["rotationAngles"])
      positioningTimeTotal += time.time()-timeBefore
      timeBefore = time.time()
      varianceImage = getVariancesAtCurrentLocation(parsedArgs.spreadSampleCount, ommatidialCount, eyeRenderer)
      renderingTimeTotal += time.time()-timeBefore
      return (np.max(varianceImage))

    # Configure bounds variables
    bboxLower = np.asarray(parsedArgs.searchBox[:3])
    bboxUpper = np.asarray(parsedArgs.searchBox[3:])
    cylinderCenter = np.asarray(parsedArgs.searchCylinder[:3])
    cylinderRadius = parsedArgs.searchCylinder[3]
    cylinderHeight = parsedArgs.searchCylinder[4]

    # Configure bounds functions to limit the search algorithm's movement
    def cylinderBounds(pose):
      diff = pose["position"] - cylinderCenter
      # Trim to a circle in XZ
      diffXZ = diff[[0,2]]
      ratio = np.sqrt(np.dot(diffXZ, diffXZ)) / cylinderRadius
      newPosition = cylinderCenter + diff*ratio
      # Trim by top and bottom
      newPositionY = max(cylinderCenter[1], min(newPosition[1], cylinderCenter[1]+cylinderHeight))
      newPosition[1] = newPositionY
      return {"position": newPosition, "rotationAngles":pose["rotationAngles"]}
    def boxBounds(pose):
      newPosition = np.maximum(bboxLower, np.minimum(pose["position"], bboxUpper))
      return {"position":newPosition, "rotationAngles":pose["rotationAngles"]}
    boundsMethods = {
      "boundless": (lambda pose: pose),
      "box": boxBounds,
      "cylinder": cylinderBounds
    }
    cullToBounds = boundsMethods[parsedArgs.boundType]

    # Actually run the genetic algorithm
    biasedChoiceDist = np.asarray([(1.0/(i+1)**2) for i in range(parsedArgs.GAgenerationSize)])
    biasedChoiceDist /= np.sum(biasedChoiceDist) # Biased list to prioritise higher scoring variations
    angularMutationScale = 0.49556443208549306 # Half the angle between two points in an icosahedron, in radians
    translationMutationScale = 100
    if parsedArgs.boundType == "box":
      translationMutationScale = np.max(bboxUpper-bboxLower)/2
    elif parsedArgs.boundType == "cylinder":
      translationMutationScale = max(cylinderRadius*2, cylinderHeight)
    mutationRate = 0.8
    startPos = np.zeros(3)

    if parsedArgs.boundType == "box":
      startPos = (bboxUpper + bboxLower)/2
    elif parsedArgs.boundType == "cylinder":
      startPos = cylinderCenter + np.asarray([0,cylinderHeight/2,0])

    poses = [{"position": startPos, "rotationAngles": (0.0,0.0,0.0)}]*parsedArgs.GAgenerationSize
    highestSpreadPose = poses[0]
    highestSpread = 0.0
    eyeRenderer.setVerbosity(False)
    beforeTime = time.time()
    print("Running genetic algorithm to find point of highest visual frequency...")
    withinSpreadCounter = 0
    for i in range(100): # Maximum of 100 iterations
      # Initiate a new generation
      lastBest = poses[0]
      poses = np.random.choice(poses, parsedArgs.GAgenerationSize, p=biasedChoiceDist) # Initial variants
      mutationMask = [np.random.random(6)<mutationRate for i in range(parsedArgs.GAgenerationSize)] # Generate a mutation mask
      poses = [{"position":p["position"]+rnd3vec(translationMutationScale)*m[:3], "rotationAngles":p["rotationAngles"]+rnd3vec(angularMutationScale)*m[3:]} for p,m in zip(poses,mutationMask)] # Mutate the poses
      poses = [cullToBounds(p) for p in poses] # Cull the poses down to those inside the defined bounds
      poses[0] = lastBest
      # Score and sortthe poses by the variation they generate
      scores = [(getMaxVarianceAtPose(p), p) for p in poses]
      scores.sort(key=lambda pair: -pair[0]) # Sorting *highest* first
      # Store the pose with the highest variation
      lastSpread = highestSpread
      highestSpreadPose = scores[0][1]
      highestSpread = scores[0][0]
      # Compare the last best pose with this one, if it's within N% of the previous one niavely assume we've reached steady state
      if abs(lastSpread - scores[0][0]) < parsedArgs.searchCutoff*highestSpread:
        withinSpreadCounter = withinSpreadCounter + 1
      else:
        withinSpreadCounter = 0

      if withinSpreadCounter > 10:
        break

      # Strip out the scores from the list of poses, just leaving the ordered poses
      poses = [p for s, p in scores]

      print("[{}] Highest variance: {}".format(i,highestSpread))
      if parsedArgs.debugVis:
        # Configure to viewable image
        eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer, originalOmmatidia)
        eyeRenderer.setCurrentEyeShaderName(c_char_p(b"spherical_orientationwise"))
        eyeRenderer.setCurrentEyeSamplesPerOmmatidium(3)
        eyeTools.setRenderSize(eyeRenderer,550,400)
        # Rotate to the highestSpreadPose
        eyeRenderer.setCameraPose(*highestSpreadPose["position"], *highestSpreadPose["rotationAngles"])
        # Render
        eyeRenderer.renderFrame()
        eyeRenderer.displayFrame()
        # Reconfigure back to raw data using only 12 ommatidium
        eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer, uniformEyeData)
        eyeRenderer.setCurrentEyeShaderName(c_char_p(b"raw_ommatidial_samples"))
        eyeRenderer.setCurrentEyeSamplesPerOmmatidium(parsedArgs.spreadSampleCount)
        eyeTools.setRenderSize(eyeRenderer, ommatidialCount, parsedArgs.spreadSampleCount)

    #Print the location and heading found as the point of maximum visual frequency
    #  This will need to be extracted by taking the ommatidium with it, retrieving it's direction (and, for results purposes, putting this into worldspace using the the eye's local coord space)
    print()
    elapsedTime = time.time()-beforeTime
    print("Highest deviation found:", highestSpread)
    print("Total elapsed time: {}s".format(elapsedTime))
    print("Positioning time  : {}s ({}%)".format(positioningTimeTotal, positioningTimeTotal/elapsedTime*100))
    print("Rendering time    : {}s ({}%)".format(renderingTimeTotal, renderingTimeTotal/elapsedTime*100))

    print("Now beginning sample spread calculation...")
    # First set it to the pose of maximum spread
    eyeRenderer.setCameraPose(*highestSpreadPose["position"], *highestSpreadPose["rotationAngles"])
    eyeRenderer.setCurrentEyeSamplesPerOmmatidium(1)
    eyeRenderer.setCurrentEyeShaderName(c_char_p(b"single_dimension_fast"))
    eyeTools.setRenderSize(eyeRenderer, ommatidialCount, 1)
    maxSd = 0
    minimumSamples = 0
    sdLimit = parsedArgs.cutoffPercent*np.linalg.norm(np.asarray([255]*3))
    standardDeviations = []
    for i in range(1,10001): # Maximum of 100000 samples
      varImage = getVarianceAtCurrentLocationWithSpecifiedSamplesPerOmm(eyeRenderer, i, parsedArgs.spreadSampleCount)
      maxSd = np.sqrt(np.max(varImage))
      standardDeviations.append(maxSd)
      if(maxSd < sdLimit):
        minimumSamples = i
        break
      print("Standard deviation at {} samples/ommatidium: {}".format(i, maxSd))

    print("Suggested minimum samples is {} samples per steradian, with a maximal variance of {}.".format(minimumSamples,maxSd))
    fig, ax = plt.subplots()
    xs = np.asarray(range(1,minimumSamples+1))
    ax.plot(xs, np.asarray(standardDeviations))
    ax.set(xlabel="rays per steradian", ylabel="Standard Deviation", title="Maximum Standard Deviation per Rays per Ommatidium")
    ax.grid()
    fig.savefig("StandardDeviationPerRaysPerSteradian.png")
    fig.show()

    eyeRenderer.stop()
  except Exception as e:
    print(e)
    if eyeRenderer != None:
      eyeRenderer.stop()

if __name__ == "__main__":
  main(sys.argv[1:])
