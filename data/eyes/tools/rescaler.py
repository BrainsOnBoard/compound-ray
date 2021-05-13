# Takes in a path to a .eye file, scales down the eye's ommatidial positions such that the average is a given size.

import sys
import math
from functools import reduce

if len(sys.argv) < 3:
  print("Usage:\npython rescaler.py <path to eye to rescale>.eye <new average ommatidial distance from eye center>")
  exit()

targetRadius = float(sys.argv[2])
outputFilepath = ".".join(sys.argv[1].split(".")[:-1]) + "-avgRadius-" + str(targetRadius) + ".eye"

with open(sys.argv[1], "r") as inputFile, open(outputFilepath, "w") as outputFile:
  # Read in the data
  lines = inputFile.readlines()
  ommatidialCount = len(lines)
  positions = []
  avgDistance = 0
  others = []
  for line in lines:
    splitLine = line.split(" ")
    position = [float(s) for s in splitLine[:3]]
    avgDistance += math.sqrt(position[0]*position[0] + position[1]*position[1] + position[2]*position[2])
    positions.append(position)
    others.append( " ".join(splitLine[3:]) )
  avgDistance /= ommatidialCount
  # Work out the scale value in order to downscale the inputs to the right size
  scaleFactor = targetRadius/avgDistance
  # Scale down and save each one
  for position, rest in zip(positions, others):
    posStrings = [str(p*scaleFactor) for p in position] # scale down and convert to string each position
    outputFile.write(" ".join(posStrings) + " " + rest)
  
