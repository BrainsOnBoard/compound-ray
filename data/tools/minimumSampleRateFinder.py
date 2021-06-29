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
  print("If you're running this from the */eye-renderer/data/tools folder, running 'export PYTHONPATH=\"$(cd ../../python-examples/ && pwd)\"' should fix it for you :)")
  exit()

from PIL import Image

def getIcoOmmatidia():
  """Returns an ommatidial array based on the points in an icosphere, so they're equidistant"""
  ## First generate the points of the icosphere
  icoPoints = []
  icoPoints.append([0,1,0]) # Top Point
  angles = [0.4 * math.pi * i for i in range(5)]
  offsetAngle = math.atan(0.5)
  # Add the upper 5 points
  planarDistance = math.cos(offsetAngle)
  verticalOffset = math.sin(offsetAngle)
  icoPoints = icoPoints + [[math.cos(a)*planarDistance, verticalOffset, math.sin(a)*planarDistance] for a in angles]
  # Add the lower 5 points
  angles = [a + 0.2 * math.pi for a in angles]
  verticalOffset *= -1
  icoPoints = icoPoints + [[math.cos(a)*planarDistance, verticalOffset, math.sin(a)*planarDistance] for a in angles]
  icoPoints.append([0,-1,0]) # Bottom point

  icoPoints = [np.asarray(p) for p in icoPoints] # Convert to numpy vectors

  ## Convert the points into an ommatidium
  # Calculate the acceptance angle for 1 steradian
  oneSteradianAcceptanceAngle = math.acos(-(1/(2*math.pi)-1)) * 2
  return [eyeTools.Ommatidium(np.zeros(3), p, oneSteradianAcceptanceAngle, 0.0) for p in icoPoints]

def getVariancesAtCurrentLocation(sampleCount, ommCount, renderer):
  samples = np.zeros((sampleCount,ommCount,3), dtype=np.uint8)
  for i in range(sampleCount):
    renderer.renderFrame()
    frameData = renderer.getFramePointer()
    #renderer.displayFrame()
    frameDataRGB = frameData[:,:,:3] # Remove the alpha component
    samples[i,:,:] = np.copy(frameDataRGB)
  avgImage = np.mean(samples, axis=0)
  differenceImages = samples - avgImage
  magnitudeImages = np.linalg.norm(differenceImages, axis=2)
  magnitudeSquaredImages = magnitudeImages * magnitudeImages
  varianceImage = np.sum(magnitudeSquaredImages, axis=0)/(sampleCount-1)
  return varianceImage

positioningTimeTotal = 0
renderingTimeTotal = 0

def main(argv):
  # Get the input parameters
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  parser.add_argument("-f, --file", type=str, required=True, metavar="FILE", nargs=1, dest="gltfPath", help="path to GlTF scene file.")
  parser.add_argument("-p, --percent", type=float, metavar="PERCENT", nargs=1, default=0.05, dest="cutoffDifference", help="The cuttoff percentage the program will increase sample count until the difference is within. 0.05 (5 percent) by default.")
  parser.add_argument("-s, --search-bound", type=str, metavar="BOUND TYPE", nargs=1, default="box", choices=["box", "cylinder"], dest="boundType", help="Configures the type of search bound to use. Options are 'box' or 'cylinder'")
  parser.add_argument("-c, --search-cylinder", type=float, default=[0,0], metavar="N", nargs=5, dest="searchCylinder", help="The specifications of a search bounding sphere - defined in the form of a single X,Y,Z coordinate for the center of the base of the cylinder and a following radius and height of the cylinder, e.g./ -c 0 0 0 2 5")
  parser.add_argument("-b, --search-box", type=float, default=[0,0,0,0,0,0], metavar="N", nargs=6, dest="searchBox", help="The coordinates - in the form of two sets of X,Y,Z coordinates (lowest, then highest), of a search bounding box, e.g./ -b 0 0 0 2 3 3")
  parser.add_argument("--lib", type=str, metavar="PATH", nargs=1, default="", dest="libPath", help="Path to the eye render shared object (.so) file. Required if this python program has been moved. Checked before default relative paths to the make and ninja buildfolders.")
  parser.add_argument("--spread-sample-count", type=int, metavar="SAMPLES", nargs=1, default=100, dest="spreadSampleCount", help="The number of images taken from a given point using the same eye configuration in order to measure standard deviation across each ommatidium.")
  parser.add_argument("--generation-size", type=int, metavar="S", nargs=1, default=100, dest="GAgenerationSize", help="Generation size of the genetic algorithm for finding the point of highest spread/visual frequency.")
  #parser.add_argument("--seed", type=int, metavar="N", nargs=1, default=42, dest="randomSeed", help="The seed passed to.... " actually, this doesn't work. GPU randoms are different on different machines.
  parsedArgs = parser.parse_args()

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
    
    print("Loading scene at {} (please wait)...".format(parsedArgs.gltfPath)[0])
    # Check the file exists
    gltfPath = pathlib.Path(parsedArgs.gltfPath[0])
    if not gltfPath.exists():
      print("Error: Supplied gltf file does not exist at " + str(gltfPath))
      exit()
    eyeRenderer.loadGlTFscene(c_char_p(str(gltfPath).encode("utf-8")))
    print("Scene loaded!")

    # Set the render size
    eyeTools.setRenderSize(eyeRenderer, 550, 400)

    # Find a compound eye in the scene, go to it.
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
      print("Error: Could not find compound eye in provided GlTF scene.")

    ### Find the point of highest visual frequency
    # Set the compound eye to a special eye design that's got, say, 100? 1000? ommatidia, equidistantly-spaced
    #   The eye will need to be using the fast vector shader :/ Can we change this on the fly? I think we can. Heck, maybe at this point we should just generate a new eye?
    #   Made using an the points on an isosphere as the axes of each from the center, cone of vision is fixed to 1sr.
    #   Resize output vector to match
    # Use simple GA (translation, small axis-angle rotation [random axis, maximum angle to angle between two points on the isosphere]) to search the space for point of max spread (highest visual freq.)
    #   Might require a "resetRotation" or "direct rotation setting" mode on the cameras :/
    #   Carry on going until the change in max variance from the previous one is below M% (Note: Different to the similar metric to use for the next step)
    # Print the location and heading found as the point of maximum visual frequency
    #   This will need to be extracted by taking the ommatidium with it, retrieving it's direction (and, for results purposes, putting this into worldspace using the the eye's local coord space)
    
    ### Actually calculate the minimum sample rate
    # (??? Is this overkill?) First, strip away all ommatidia but the one with the maximum frequency (omm[maxfreqID])
    #   omm[maxFreqID]
    #   resize output
    # Increase samples per ommatidium until the increase spread (which will be per-steradian, because each of the cones will be 1sr) isn't changing by more than N% (configured) of it's previous
    # Bam! You have the minimum samples per steradian per ommatidium to configure your experiments with.
    # Indicate to the user that they can get the correct samples per ommatidium required for a given scene by running <ANOTHER TOOL HERE>
    #   This tool will go through every compound eye in the provided scene, and find the maximum solid angle of any compound eye (in steradians), then multiply the provided samples-per-omm value by it
    #   Alternatively, if a camera name is provided, it'll do it only for that eye.

    ### Find the point of highest visual frequency
    ## Set the compound eye to a special design that's got a set number of equidistantly-spaced ommatidia.
    eyes = getIcoOmmatidia()
    ommatidialCount = len(eyes)
    eyeTools.setOmmatidiaFromOmmatidiumList(eyeRenderer, eyes)
    ## Configure the eye to output a single dimension, and resize the output
    eyeRenderer.setCurrentEyeShaderName(c_char_p(b"single_dimension_fast"))
    eyeTools.setRenderSize(eyeRenderer,ommatidialCount,1)
    ## Use simple GA (translation, small axis-angle rotation [random axis, maximum angle to angle between two points on the isosphere]) to search the space for point of max spread (highest visual freq.)
    global positioningTimeTotal
    global renderingTimeTotal
    def biasedChoice(ls):
      """Chooses from the given list, giving more weight to the first"""
      worstSelectionProb = 1.0/len(ls)
      selectionID = int(1/(np.random.uniform()*(1.0-worstSelectionProb)+worstSelectionProb)) -1
      return ls[selectionID]
    def rnd3vec(scale):
      """Returns a random 3vector with values between -scale and +scale"""
      return np.random.uniform(-1,1,size=3)*scale
    def getMaxVarianceAtPose(pose):
      """Moves a the camera to the pose, then calculates the maximum variance at that location from all variances at all eyes"""
      # First reset the camera, and then move it to the pose
      global positioningTimeTotal
      global renderingTimeTotal
      timeBefore = time.time()
      #eyeRenderer.resetCameraPose()
      #eyeRenderer.rotateCameraAround(pose["rotationAngles"][0] ,1,0,0)
      #eyeRenderer.rotateCameraAround(pose["rotationAngles"][1] ,0,1,0)
      #eyeRenderer.rotateCameraAround(pose["rotationAngles"][2] ,0,0,1)
      #eyeRenderer.translateCamera(*pose["position"])
      eyeRenderer.setCameraPose(*pose["position"], *pose["rotationAngles"])
      positioningTimeTotal += time.time()-timeBefore
      timeBefore = time.time()
      varianceImage = getVariancesAtCurrentLocation(parsedArgs.spreadSampleCount, ommatidialCount, eyeRenderer)
      renderingTimeTotal += time.time()-timeBefore
      return (np.max(varianceImage))
    ##   Might require a "resetRotation" or "direct rotation setting" mode on the cameras :/
    ##   Carry on going until the change in max variance from the previous one is below M% (Note: Different to the similar metric to use for the next step)
    angularMutationScale = 0.49556443208549306 # Half the angle between two points in an icosahedron, in radians
    translationMutationScale = 5 # Probably should be some percentage of the total area of the volume or something
    mutationRate = 0.8
    poses = [{"position": np.zeros(3), "rotationAngles": (0.0,0.0,0.0)}]
    highestSpreadPose = poses[0]
    highestSpread = 0.0
    eyeRenderer.setVerbosity(False)
    beforeTime = time.time()
    for i in range(10):
      # Initiate a new generation
      poses = [biasedChoice(poses) for i in range(parsedArgs.GAgenerationSize)] # Initial variants
      mutationMask = [np.random.random(6)<mutationRate for i in range(parsedArgs.GAgenerationSize)] # Generate a mutation mask
      poses = [{"position":p["position"]+rnd3vec(translationMutationScale)*m[:3], "rotationAngles":p["rotationAngles"]+rnd3vec(angularMutationScale)*m[3:]} for p,m in zip(poses,mutationMask)] # Mutate the poses
      # Score and sortthe poses by the variation they generate
      scores = [(getMaxVarianceAtPose(p), p) for p in poses]
      scores.sort(key=lambda pair: pair[0])
      # Store the pose with the highest variation
      highestSpreadPose = scores[0][1]
      highestSpread = scores[0][0]
      # Strip out the scores from the list of poses, just leaving the ordered poses
      poses = [p for s, p in scores]

      print("[{}] Highest variance: {}".format(i,highestSpread))

    #Print the location and heading found as the point of maximum visual frequency
    #  This will need to be extracted by taking the ommatidium with it, retrieving it's direction (and, for results purposes, putting this into worldspace using the the eye's local coord space)
    print()
    elapsedTime = time.time()-beforeTime
    print("Total elapsed time: {}s".format(elapsedTime))
    print("Positioning time  : {}s ({}%)".format(positioningTimeTotal, positioningTimeTotal/elapsedTime*100))
    print("Rendering time    : {}s ({}%)".format(renderingTimeTotal, renderingTimeTotal/elapsedTime*100))

    eyeRenderer.stop()
  except Exception as e:
    print(e)

if __name__ == "__main__":
  main(sys.argv[1:])


    #eyeRenderer.setCurrentEyeSamplesPerOmmatidium(300)
    #eyeRenderer.renderFrame()
    #eyeRenderer.displayFrame()
    #time.sleep(2)

    #eyeRenderer.resetCameraPose()
    #eyeRenderer.renderFrame()
    #eyeRenderer.displayFrame()
    #time.sleep(2)

    #moveAngle = 3
    #speed = -0.1
    #for i in range(int(360/moveAngle)):
    #  #eyeRenderer.rotateCameraAround(moveAngle/180*math.pi, 1, 0, 0)
    #  eyeRenderer.translateCamera(0, 0, speed)
    #  eyeRenderer.renderFrame()
    #  eyeRenderer.displayFrame()
    #  time.sleep(0.1)







    #SAMPLES_PER_OMMATIDIUM = 1
    #eyeRenderer.setCurrentEyeSamplesPerOmmatidium(SAMPLES_PER_OMMATIDIUM)

    #ommatidialCount = eyeRenderer.getCurrentEyeOmmatidialCount()
    #eyeTools.setRenderSize(eyeRenderer, 400,400)
    #eyeRenderer.setCurrentEyeShaderName(c_char_p(b"spherical_orientationwise"))
    #eyeRenderer.renderFrame()
    #eyeRenderer.displayFrame()
    #eyeRenderer.saveFrameAs(c_char_p("{}-regularImg.ppm".format(SAMPLES_PER_OMMATIDIUM).encode()))

    #eyeRenderer.setCurrentEyeShaderName(c_char_p(b"spherical_orientationwise_ids"))
    #eyeRenderer.renderFrame()
    #idMap = np.flipud(np.copy(eyeRenderer.getFramePointer()))

    #eyeRenderer.setCurrentEyeShaderName(c_char_p(b"single_dimension_fast"))
    #eyeRenderer.renderFrame()
    #eyeRenderer.displayFrame()

    #eyeTools.setRenderSize(eyeRenderer, ommatidialCount, 1)
    #varianceImage = getVariancesAtCurrentLocation(parsedArgs.spreadSampleCount, ommatidialCount, eyeRenderer)
    #print()
    #print("MAXIMUM VARIANCE: {}".format(np.max(varianceImage)))
    #varianceImage = (varianceImage/np.max(varianceImage)) * 255 # normalise between 0 and 255.
    ##varianceImage = (varianceImage/1680.926) * 255 # normalise between 0 and 255.
    #projectionImage = eyeTools.getProjectionImageUsingMap(varianceImage, idMap, 400, 400)
    #Image.fromarray(projectionImage, mode="L").save("{}-projection.png".format(SAMPLES_PER_OMMATIDIUM))

