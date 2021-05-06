# splits an eye along the x-axis
# Position, orientation, acceptance angle (degrees)

import os
import sys
import getopt
import numpy as np

def main(argv):
  USAGE_STRING = "USAGE:\n\npython eyeSplitter.py -f <file containing eye to split> -d <distance to split> -o <overlap (0-1)>"
  
  infilePath = None
  distance = None
  overlap = None
  
  try:
    opts, args = getopt.getopt(argv, "hf:d:o:")
  except getopt.GetoptError:
    print(USAGE_STRING)

  for opt, arg in opts:
    if opt == "-h":
      print(USAGE_STRING)
      sys.exit()
    elif opt == "-f":
      infilePath = arg
    elif opt == "-d":
      distance = float(arg)
    elif opt == "-o":
      overlap = float(arg)

  if infilePath == None or distance == None or overlap == None:
    print(USAGE_STRING)
    sys.exit()

  outfilePath = ".".join(infilePath.split(".")[:-1]) + "-" + str(distance) + "-" + str(overlap) + ".eye"

  with open(infilePath, "r") as infile, open(outfilePath, "w") as outfile:
    leftShift = np.asarray([-distance, 0, 0])
    rightShift = leftShift*-1
    for line in infile:
      splitLine = line.split(" ")
      splitFloats = [float(n) for n in line.split(" ")]
      position = np.asarray(splitFloats[:3])
      # If it's on the left
      if position[0] < overlap:
        leftPos = position + leftShift
        leftPos = [str(n) for n in leftPos]
        outfile.write(" ".join(leftPos) + " " + " ".join(splitLine[3:]))
      # If it's on the right
      if position[0] > -overlap:
        rightPos = position + rightShift
        rightPos = [str(n) for n in rightPos]
        outfile.write(" ".join(rightPos) + " " + " ".join(splitLine[3:]))


if __name__ == "__main__":
  main(sys.argv[1:])
