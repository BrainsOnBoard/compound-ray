# This file is used to generate graphs showing the learning rate of all eye variants of each of the eye designs.
# By running it, the data will be loaded from the `data-out` folder for the loss graph of each eye design and
# displayed here one at a time, before finally showing all graphs in one.
# `data-out` needs to be populated with loss graphs, generated using position-estimator-file-based.py
#
# By changing the types and eyeNames specified in their respective variables, you can change which loss graphs are loaded.


import numpy as np
import matplotlib.pyplot as plt

types = ["real", "single", "split"]
colours = {"real": "red", "single":"blue", "split":"black"}
eyeNames = ["AM_60185", "AM_60186", "BT_77966", "BT_77967", "BT_77970", "BT_77971", "BT_77973", "BT_77974"]
testSamplingSize = 100
maxEpochs = 100
splitPct = 0.8

xs = np.arange(maxEpochs) + 1

# Individual:
for eyeName in eyeNames:
  for dataType in types:
    graphPath = f"data-out/LossGraph-{eyeName}-{dataType}-{testSamplingSize}grid-{maxEpochs}epochs-{splitPct}splitPct.npy"
    data = np.load(graphPath)
    plt.plot(xs, data, label = dataType)#, color=colours[dataType])
  plt.legend()
  plt.xlabel("Epochs")
  plt.ylabel("Error")
  plt.title(f"{eyeName}: Validation Error During Training")
  plt.show()

#plt.legend()
#plt.xlabel("Epochs")
#plt.ylabel("Error")
#plt.title("Validation Error During Training")
#plt.show()

# All on one:
for eyeName in eyeNames:
  for dataType in types:
    graphPath = f"data-out/LossGraph-{eyeName}-{dataType}-{testSamplingSize}grid-{maxEpochs}epochs-{splitPct}splitPct.npy"
    data = np.load(graphPath)
    plt.plot(xs, data, label = dataType, color=colours[dataType])
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title(f"All Eyes: Validation Error During Training")
plt.show()


# Diff plot:
for eyeName in eyeNames:
  for dataType in types:
    graphPath = f"data-out/LossGraph-{eyeName}-{dataType}-{testSamplingSize}grid-{maxEpochs}epochs-{splitPct}splitPct.npy"
    data = np.load(graphPath)
    plt.plot(xs, data, label = dataType, color=colours[dataType])
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title(f"All Eyes: Validation Error During Training")
plt.show()

# 
