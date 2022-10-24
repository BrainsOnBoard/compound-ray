# This script compares error volumes exported from `position-estimator-file-based.py` and stored in `data-out`.

import numpy as np
import argparse
import plotly.graph_objects as go
import numpy as np
import sys

#main(sys.argv[1:])

parser = argparse.ArgumentParser(description="Compares volumes.")

X, Y, Z = np.mgrid[-25:25:100j, -25:25:100j, -25:25:100j]

eyeName = "AM_60185"
if(len(sys.argv) >= 2):
  eyeName = sys.argv[1]

lossVolumes = {}
for dt in ["real", "split", "single"]:
  lossVolumes[dt] = np.load(f"data-out/LossVolume-{eyeName}-{dt}-100grid-100epochs-0.8splitPct.npy")

# Display the loss volumes
for graphTitle in lossVolumes.keys():
  fig = go.Figure(data=go.Volume(
      x=X.flatten(),
      y=Y.flatten(),
      z=Z.flatten(),
      value=lossVolumes[graphTitle].flatten(),
      isomin=0.1,
      isomax=0.8,
      opacity=0.1, # needs to be small to see through all surfaces
      surface_count=17, # needs to be a large number for good volume rendering
      ))
  fig.update_layout(
    title=graphTitle
  )
  fig.show()

# Display the loss differnetials
diffVolumes = {}
#diffVolumes["realVsSplit"] = np.abs(lossVolumes["real"]-lossVolumes["split"])
#diffVolumes["realVsSingle"] = np.abs(lossVolumes["real"]-lossVolumes["single"])
#diffVolumes["singleVsSplit"] = np.abs(lossVolumes["single"]-lossVolumes["split"])
diffVolumes["realVsSplit"] = lossVolumes["real"]-lossVolumes["split"]
diffVolumes["realVsSingle"] = lossVolumes["real"]-lossVolumes["single"]
diffVolumes["singleVsSplit"] = lossVolumes["single"]-lossVolumes["split"]

for graphTitle in diffVolumes.keys():
  fig = go.Figure(data=go.Volume(
      x=X.flatten(),
      y=Y.flatten(),
      z=Z.flatten(),
      value=diffVolumes[graphTitle].flatten(),
      isomin=0.1,
      isomax=0.8,
      opacity=0.1, # needs to be small to see through all surfaces
      surface_count=17, # needs to be a large number for good volume rendering
      ))
  fig.update_layout(
    title=graphTitle
  )
  fig.show()
