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

def getIdFromMap(mapImage, x, y):
  r = mapImage[y,x,0] << 24 # Red
  g = mapImage[y,x,1] << 16 # Green
  b = mapImage[y,x,2] <<  8 # Blue
  a = mapImage[y,x,3]       # Alpha
  idOut = r | g | b | a
  return(idOut)

def getProjectionImageUsingMap(vector, vectorMax, idMap, pjWidth,pjHeight):
  np.copy(idMap)
  output = np.zeros((pjWidth, pjHeight), dtype=np.uint8)
  for x in range(pjWidth):
    for y in range(pjHeight):
      pixelId = getIdFromMap(idMap, x, y)
      output[x,y] = int(vector[pixelId]/vectorMax * 255)
  return(output)

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

  # Make sure there's a place to save to
  Path("output/generated-data/alias-demo-quantified/").mkdir(parents=True, exist_ok=True)
  Path("output/vector-data/").mkdir(parents=True, exist_ok=True)
  Path("output/view-images/").mkdir(parents=True, exist_ok=True)
  Path("output/generated-data/spread-analysis/").mkdir(parents=True, exist_ok=True)
  
  ###### First, generate the ommatidial id map

  #Resize the renderer display in order to render the spherically-projected variable sample rate
  renderWidth = 700
  renderHeight = 300
  eyeTools.setRenderSize(eyeRenderer, renderWidth, renderHeight)

  # Go to the 'insect-eye-spherical-projector' camera
  eyeRenderer.gotoCameraByName(c_char_p(b"insect-eye-spherical-projector-ids"))

  eyeRenderer.renderFrame() # Just straight-up render the spherical projector ids
  idMap = np.copy(np.flipud(eyeRenderer.getFramePointer())) # Copy (remember here the data is still owned by the render, so we need this copy) the id map (plus flip it the right way up)
  eyeRenderer.saveFrameAs(c_char_p(("output/generated-data/alias-demo-quantified/projection-ids.ppm").encode())) # Save the image for sanity checking

  # Also generate a set of weights that store how much of an influence on an
  # average each compound eye should have based on it's area coverage in steradians
  perSteradianWeights = [1.0/i.getSolidAngle() for i in eyeTools.readEyeFile("../../data/eyes/1000-horizontallyAcute-variableDegree.eye")]
  perSteradianWeights = np.asarray(perSteradianWeights)

  ###### Second, generate ommatidial sample data into a big multi-dim array to perform analysis on
  
  # Change to vector rendering
  eyeRenderer.gotoCameraByName(c_char_p(b"insect-eye-fast-vector"))

  # Prepare to generate vector data (1000 ommatidia)
  vectorWidth = 1000
  maxOmmatidialSamples = renderWidth # The upper bound of how many samples will be taken per ommatidium in the analysis
  spreadSampleCount = 1000 # How many times each frame is rendered to get a sense of the spread of results from a given ommatidium at different sampling rates
  eyeTools.setRenderSize(eyeRenderer, vectorWidth, 1)

  # Create a numpy array to store the eye data
  # This is a set of eye matricies, each one being a 1st-order stack of samples (the width of the number of ommatidia, and 3 channels deep)
  eyeSampleMatrix = np.zeros((maxOmmatidialSamples,spreadSampleCount, vectorWidth, 3), dtype=np.uint8)

  # Iterate over eye sample counts
  for idx, samples in enumerate(range(1, maxOmmatidialSamples+1)):
    eyeRenderer.setCurrentEyeSamplesPerOmmatidium(samples)
    eyeRenderer.renderFrame() # First call to ensure randoms are configured

    # For each sample count, generate N images to compare
    for i in range(spreadSampleCount):
      renderTime = eyeRenderer.renderFrame() # Second call to actually render the image

      # Retrieve the data
      frameData = eyeRenderer.getFramePointer()
      frameDataRGB = frameData[:,:,:3] # Remove the alpha channel
      eyeSampleMatrix[idx,i,:,:] = np.copy(frameDataRGB[:, :, :])

    eyeRenderer.displayFrame()

  ###### Now calculate the per-ommatidial sample variance and standard deviation at each sample rate
  print("")
  maxSd = 0
  maxVariance = 0
  variances = np.zeros((renderWidth, vectorWidth, 1))
  standardDeviations = np.zeros((renderWidth, vectorWidth, 1))
  avgVariancePerSteradianPerImage = np.zeros(renderWidth)
  avgSdPerSteradianPerImage = np.zeros(renderWidth)
  for sampleCount, ommatidialSamples in enumerate(eyeSampleMatrix):
    # Get the per-ommatidial spread here
    meanImage = np.mean(ommatidialSamples, axis=0) # The means of each ommatidium (RGB)
    summedSquaredDifferences = np.zeros((meanImage.shape[0],1))
    for ommatidiumId, image in enumerate(ommatidialSamples):
      print("\rCalculating per-ommatidial spread at each sample rate (Image: {}, Sample: {})...".format(sampleCount+1, ommatidiumId+1), end='')
      for indexInMean, pixelInImage in enumerate(image):
        difference = np.linalg.norm(pixelInImage - meanImage[indexInMean,:])
        difference = difference*difference
        summedSquaredDifferences[indexInMean] += difference
    varianceImage = summedSquaredDifferences / (len(ommatidialSamples)-1)
    sdImage = np.sqrt(varianceImage)
    variances[sampleCount] = varianceImage
    standardDeviations[sampleCount] = sdImage
    # Keep track of the maximum variance and standard deviation
    maxVariance = max(maxVariance, np.max(varianceImage))
    maxSd = max(maxSd, np.max(sdImage))

    # Calculate and store the average per-steradian variance and standard deviation for this image
    avgVariancePerSteradianPerImage[sampleCount] = np.mean(varianceImage * perSteradianWeights)
    avgSdPerSteradianPerImage[sampleCount] = np.mean(sdImage * perSteradianWeights)

    # Save the image data for later use
    np.savetxt("output/vector-data/variance-"+str(sampleCount)+"-samples.txt", varianceImage, delimiter=",")
    np.savetxt("output/vector-data/sd-"+str(sampleCount)+"-samples.txt", varianceImage, delimiter=",")
  # Save the per-steradian per-image variance and sd
  np.savetxt("output/vector-data/avgPerImagePerSteradianVariance(0-{}samples).txt".format(renderWidth), avgVariancePerSteradianPerImage, delimiter=",")
  np.savetxt("output/vector-data/avgPerImagePerSteradianSd(0-{}samples).txt".format(renderWidth), avgSdPerSteradianPerImage, delimiter=",")

  # Now loop over all the generated data and create graphical
  # plots of the sd & variance using theid map.
  combinedVarImage = np.zeros((renderHeight, renderWidth), dtype=np.uint8)
  combinedSdImage = np.zeros((renderHeight, renderWidth), dtype=np.uint8)
  print("")
  for imageNumber, (varianceImage, standardDeviationImage) in enumerate(zip(variances, standardDeviations)):
    # Generate a graphical plot of the standard deviation using the id map
    print("\rGenerating mapped graphical plot of sd and variance for image {} using {} samples...".format(imageNumber+1,imageNumber+1), end='')
    sdProjectionImage = getProjectionImageUsingMap(standardDeviationImage, maxSd, idMap, renderWidth, renderHeight)
    sdProjectionImage = np.transpose(sdProjectionImage)
    combinedSdImage[:, imageNumber] = sdProjectionImage[:, imageNumber]
    Image.fromarray(sdProjectionImage, mode="L").save("output/generated-data/spread-analysis/standardDeviation-"+str(imageNumber)+"-samples.png")

    varProjectionImage = getProjectionImageUsingMap(varianceImage, maxVariance, idMap, renderWidth, renderHeight)
    varProjectionImage = np.transpose(varProjectionImage)
    combinedVarImage[:, imageNumber] = varProjectionImage[:, imageNumber]
    Image.fromarray(varProjectionImage, mode="L").save("output/generated-data/spread-analysis/variance-"+str(imageNumber)+"-samples.png")

  print("")

  eyeRenderer.stop() # Stop the renderer
  print("Final averaged per-steradian variance, per image:")
  print(avgVariancePerSteradianPerImage)
  print("Final averaged per-steradian standard deviation, per image:")
  print(avgSdPerSteradianPerImage)
  
  # Finally save the combined image
  print("Saving combined images...")
  Image.fromarray(combinedVarImage, mode="L").save("output/view-images/combined-variance-0-"+str(renderWidth)+"samples.png")
  Image.fromarray(combinedSdImage, mode="L").save("output/view-images/combined-standard-deviation-0-"+str(renderWidth)+"samples.png")

except Exception as e:
  print(e)
