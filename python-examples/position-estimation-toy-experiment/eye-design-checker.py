# This is a utility script which uses Open3D to visualise the eye design files, displaying
# each variation of a specified eye design (eyeName) as a pointcloud of it's ommatidia.

import numpy as np # For manipulation
import open3d as o3d # For visualisation
import functools

import eyeRendererPaths
import sys
from ctypes import *
sys.path.insert(1, eyeRendererPaths.PYTHON_EXAMPLES_PATH)
import eyeRendererHelperFunctions as eyeTools

eyeName = "AM_60185"

colours = {"real": [1,0,0],
           "split": [0,1,0],
           "single": [0,0,1]
          }

pointClouds = {}
for fileType in ["real", "split", "single"]:
  # Load ommatidia
  omms = eyeTools.readEyeFile(f"eye-data/{eyeName}-{fileType}.eye")

  # Convert to numpy array
  points = np.asarray(list(map(lambda n: n.position, omms)))

  # Put them into a point cloud
  pointClouds[fileType] = o3d.geometry.PointCloud()
  pointClouds[fileType].points = o3d.utility.Vector3dVector(points)
  pointClouds[fileType].normals = o3d.utility.Vector3dVector(np.asarray(list(map(lambda n: n.direction, omms))))
  pointClouds[fileType].paint_uniform_color(colours[fileType])

# Display
o3d.visualization.draw_geometries([pointClouds["real"], pointClouds["split"], pointClouds["single"]])
#o3d.visualization.draw_geometries([pointClouds["real"], pointClouds["single"]])
#o3d.visualization.draw_geometries([pointClouds["real"]])
#o3d.visualization.draw_geometries([pointClouds["split"]])
#o3d.visualization.draw_geometries([pointClouds["single"]])
