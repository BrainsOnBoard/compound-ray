DESCRIPTION = "Allows you to set the center point of an .obj object."

import numpy as np
import getopt
import argparse
import sys
import os

def main(argv):
  # Defaults
  inputPaths = None
  outputName = "centered"
  translation = np.asarray([0,0,0])

  # Get the input parameters
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  parser.add_argument("-f, --file", type=str, required=True, metavar="FILE", nargs='+', dest="paths", help="path(s) to input files.")
  parser.add_argument("-s, --output-suffix", type=str, default="centered", metavar="SUFFIX", nargs=1, dest="suffix", help="appends SUFFIX to output filenames to differentiate between in and out files.")
  parser.add_argument("-t, --translation", type=float, default=[0,0,0], metavar="T", nargs=3, dest="translation", help="translates the obj by vector [T,T,T].")
  parser.add_argument("-m, --move-only", type=bool, dest='moveOnly', default=False, help="switch to disable centering. will only translate the object(s) by [T,T,T] supplied via --translation.")

  parsedArgs = parser.parse_args()
  translation = np.asarray(parsedArgs.translation)

  print(str(len(parsedArgs.paths)) + " files selected.")

  # Calculate the average center point of all vertices
  pointCount = 0
  avgPos = np.zeros(3)
  for fpath in parsedArgs.paths:
    with open(fpath, 'r') as f:
      prevPointCount = pointCount # Used for debug output to get per-obj vert count
      prevAvgPos = avgPos.copy()

      for line in f.readlines():
        splitData = line.strip().split(" ")
        if not (len(splitData) > 0 and splitData[0] == "v"):
          continue # Skip this line as it's not a vertex line (desparately avoiding indents)
        vertPos = np.asarray([float(n) for n in splitData[1:4]])
        avgPos += vertPos
        pointCount = pointCount+1 
       
      objPointCount = pointCount - prevPointCount
      objAvg = avgPos - prevAvgPos
      print("\nFile Path: %s\n\tVertices:%i\n\tAvgPos:%s"%(fpath,objPointCount,str(objAvg/objPointCount)))

  avgPos /= pointCount # Calculate the average position of all points from all files
  print("\nTotal vertices processed:%i\nAverage position:%s"%(pointCount, avgPos))
  print("Saving files...\n")

  for inPath in parsedArgs.paths:
    outPath = inPath[:-4] + "-" + parsedArgs.suffix + ".obj"
    with open(inPath, 'r') as inf, open(outPath, 'w') as outf:
      outf.write("# Object has been recentered by translation through -" + str(avgPos) + " + " + str(translation))
      for line in inf.readlines():
        splitData = line.strip().split(" ")
        if not (len(splitData) > 0 and splitData[0] == "v"):
          # This line isn't a vertex line, just write it directly to the new file
          outf.write(line)
        else:
          # Extract the vertex position and center + translate it
          vertPos = np.asarray([float(n) for n in splitData[1:4]])
          # Center
          if not parsedArgs.moveOnly:
            vertPos = vertPos - avgPos
          # Translate it
          vertPos += translation
          
          # Write the start of the line
          outf.write(splitData[0] + " ")
          # Write the translated vertex back, including the 
          outf.write(" ".join(["{:.6f}".format(n) for n in vertPos]) + " " + " ".join(splitData[4:]) + os.linesep)
      print("Written file " + outPath + ".")

if __name__ == "__main__":
  main(sys.argv[1:])
