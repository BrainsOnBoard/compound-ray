# Similar to `all-learning-graphs.py` but renders just one eye type's performance across it's split, single
# and real derivatives. Ensure that `data-out` is populated with loss graphs before running, generated using
# position-estimator-file-based.py

import numpy as np
import matplotlib.pyplot as plt

eyeName = "AM_60185"
types = ["real", "single", "split"]
testSamplingSize = 100
maxEpochs = 100
splitPct = 0.8

xs = np.arange(maxEpochs) + 1

for dataType in types:
  graphPath = f"data-out/LossGraph-{eyeName}-{dataType}-{testSamplingSize}grid-{maxEpochs}epochs-{splitPct}splitPct.npy"
  data = np.load(graphPath)
  plt.plot(xs, data, label = dataType)

plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Validation Error During Training")
plt.show()
