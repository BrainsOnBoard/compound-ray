# Recalculates the acceptance angles of each ommatidium based on the angular distance to the ommatidium closest to it.
# Assumes that this is a spherical eye.

import os
import sys
import getopt
import numpy as np
import math

angleBetween = lambda v1, v2: math.acos(v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

def main(argv):
  USAGE_STRING = "USAGE:\n\npython acceptanceAngleAdjuster.py -f <file containing eye to adjust>"
  
  infilePath = None
  
  try:
    opts, args = getopt.getopt(argv, "hf:")
  except getopt.GetoptError:
    print(USAGE_STRING)
	sys.exit()

  for opt, arg in opts:
    if opt == "-h":
      print(USAGE_STRING)
      sys.exit()
    elif opt == "-f":
      infilePath = arg

  if infilePath == None:
    print(USAGE_STRING)
    sys.exit()

  outfilePath = ".".join(infilePath.split(".")[:-1]) + "-maximalAcceptanceAngles.eye"

  with open(infilePath, "r") as infile, open(outfilePath, "w") as outfile:
    directions = []
    closests = []
    maximalRadii = []
    lineBeginnings = []
    lineEnds = []
    for line in infile:
      splitLine = line.split(" ")
      splitFloats = [float(n) for n in line.split(" ")]
      directions.append(np.asarray(splitFloats[3:6]))
      lineBeginnings.append(" ".join(splitLine[:6])+" ")

    # Get the closest direction
    for dirIdx, direction in enumerate(directions):
      closestDistance = float("inf")
      closest = 0
      for index, otherDirection in enumerate(directions):
        if(index == dirIdx): # Don't compare the same index
          continue
        angularDistance = angleBetween(direction, otherDirection)
        if(angularDistance < closestDistance):
          closestDistance = angularDistance
          closest = index
      closests.append(closest)
      maximalRadii.append(closestDistance)

    # Now create a new list of maximal radii
    trueMaximalAcceptanceAngles = []
    for index, direction in enumerate(directions):
      trueMaximalAcceptanceAngles.append( (maximalRadii[index]-maximalRadii[closests[index]]/2.0) * 2 )

    # Write to the new file
    for beginning, acceptanceAngle in zip(lineBeginnings, trueMaximalAcceptanceAngles):
      outfile.write(beginning + str(acceptanceAngle) + "\n")


if __name__ == "__main__":
  main(sys.argv[1:])
