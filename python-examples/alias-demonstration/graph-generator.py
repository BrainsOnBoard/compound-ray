import numpy as np
import matplotlib.pyplot as plt

logScale = True

averageSamples = np.loadtxt("output/vector-data/avgPerImagePerSteradianSd(0-700samples).txt")
sampleIndices = np.arange(1,701)
plt.title("Average Standard Deviation per Steradian per Ommatidial Sample Count")
plt.xlabel("Number of Samples per Ommatidium")
plt.ylabel("Average Standard Deviation per Steradian")

if(logScale):
  plt.xscale("log")

plt.plot(sampleIndices, averageSamples)
plt.show()
