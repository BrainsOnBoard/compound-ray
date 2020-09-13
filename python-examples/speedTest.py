import time
from ctypes import *
from sys import platform
import numpy as np
from matplotlib import pyplot as plt

import eyeRendererHelperFunctions as eyeTools


outputFile = open("outputData-speed-ofstad-100omm,300frameAvg,0-10000(10)samples.csv", "w")
outputFile.write("#samples per eye, samples total, fps\n")
try:
  eyeRenderer = CDLL("/home/blayze/Documents/new-renderer/build/ninja/lib/libEyeRenderer3.so")
  print("Successfully loaded ", eyeRenderer)

  # Configure the renderer's function outputs and inputs:
  eyeTools.configureFunctions(eyeRenderer)

  eyeRenderer.loadGlTFscene(c_char_p(b"/home/blayze/Documents/new-renderer/data/ofstad-arena/ofstad-speed-test-1.gltf"))

  numberOfOmmatidia = 100

  eyeTools.setRenderSize(eyeRenderer, numberOfOmmatidia, 1)

  eyeRenderer.setVerbosity(False) # Turn off verbose messaging for speed
  
  results = []
  samplesPerOmmAxis = []
  speedAxis = []
  numberOfSampleFramesToRender = 300
  for i in range(0, 10001, 10):
    eyeTools.setSamplesPerOmmatidium(eyeRenderer, i)

    # Average the render time of `numberOfSampleFramesToRender` frames
    avgFrameRenderTime = 0.0
    for o in range(numberOfSampleFramesToRender):
      renderTime = eyeRenderer.renderFrame()
      avgFrameRenderTime += renderTime
      #eyeRenderer.displayFrame()
    avgFrameRenderTime /= numberOfSampleFramesToRender

    samples = i*numberOfOmmatidia
    fps = 1000.0/avgFrameRenderTime
    samplesPerOmmAxis.append(samples)
    speedAxis.append(fps)
    outputFile.write(str(i) + ", " + str(samples) +  ", " + str(fps) + "\n")

  plt.title("Speed")
  plt.xlabel("Total scene samples ("+str(numberOfOmmatidia)+" ommatidia)")
  plt.ylabel("fps")
  plt.plot(samplesPerOmmAxis, speedAxis)
  plt.show()


  eyeRenderer.stop()
except Exception as e:
  print(e);

outputFile.close()

