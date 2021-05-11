# Orthogonally renders a given eye to an SVG file
# Uses svgwrite: https://github.com/mozman/svgwrite

import sys
import svgwrite
import numpy as np
import getopt
import math

TOWARD_CAMERA = np.asarray([0,0,-1])

def helpAndExit(error = None):
  USAGE_STRING = """
USAGE:
python eyeVisualiser.py -f <eye file path> 

Optionals:
-y <yaw>\thorizontal yaw (azimuth) of the camera , indegrees
-p <pitch>\tvertical pitch (altitude) of the camera, in degrees
-o <outpath>\toutput image path (svg, png)
-n <length>\tfacet normal length (same units as the eye model)
-s <scale>\timage scale (default: 1mm = 1 unit SVG scale)
-b \t\tSwitch to enable rendering of away-facing ommatidia"""
  if error:
    print("Error:", error)
  print(USAGE_STRING)
  sys.exit()

def getEyeFeatures(line):
  data = [float(n) for n in line.split(" ")]
  position = np.asarray(data[:3])
  direction = np.asarray(data[3:6])
  acceptanceAngle = data[6]
  focalPointOffset = data[7]
  return (position, direction, acceptanceAngle, focalPointOffset)
  
def draw3Dline(pos, extent, svg):
  startX = pos[0] # pos.x
  startY = pos[1] # pos.y
  end = pos + extent
  endX = end[0]
  endY = end[1]
  svg.add(svg.line((startX, startY), (endX, endY), stroke=svgwrite.rgb(10,10,16,'%')))

def main(argv):
  # Defaults
  infilePath = None
  yaw = 0
  pitch = 0
  outputPath = "visualisedEye.svg"
  renderBackfaces = False
  normalLength = 1
  imgScale = 1
  
  # Get the input parameters
  try:
    opts, args = getopt.getopt(argv, "f:y:p:w:h:o:bn:s:")
  except getopt.GetoptError:
    helpAndExit()
  
  for opt, arg in opts:
    if opt == "--help" or opt == "-h":
      helpAndExit()
    elif opt == "-f":
      infilePath = arg
    elif opt == "-y":
      yaw = float(arg)
    elif opt == "-p":
      pitch = float(arg)
    elif opt == "-o":
      outputPath = arg
    elif opt == "-b":
      renderBackfaces = True
    elif opt == "-n":
      normalLength = float(arg)
    elif opt == "-s":
      imgScale = float(arg)
    
  if infilePath == None:
    helpAndExit("Please provide an input eye filepath.")

  # Convert angles into radians
  yaw = yaw/180.0 * math.pi
  pitch = pitch/180.0 * math.pi
  
  # Create the image
  dwg = svgwrite.Drawing(outputPath, profile='tiny')
  
  # Read and render the eye
  with open(infilePath, "r") as eyeFile:
    for line in eyeFile:
      # Extract data
      ommatidialPosition, ommatidialAxis, acceptance, offset = getEyeFeatures(line)
      
      # Scale normal
      ommatidialAxis *= normalLength

      # Scale the entire image
      ommatidialPosition *= imgScale
      ommatidialAxis *= imgScale

      # Rotate the ommatidial axis line
      # x-y plane
      yawRotMatrix = np.asarray([[math.cos(yaw) , 0, math.sin(yaw)],
                                 [0             , 1,             0],
                                 [-math.sin(yaw), 0, math.cos(yaw)]])

      # y-z plane
      pitchRotMatrix = np.asarray([[1,              0 ,                0],
                                   [0, math.cos(pitch), -math.sin(pitch)],
                                   [0, math.sin(pitch), math.cos(pitch)]])

      rotationMatrix = np.matmul(pitchRotMatrix, yawRotMatrix)

      ommatidialPosition = rotationMatrix.dot(ommatidialPosition)
      ommatidialAxis = rotationMatrix.dot(ommatidialAxis)

      # Cull back-facing ommatidia
      if not renderBackfaces and np.dot(ommatidialAxis, TOWARD_CAMERA) < 0:
        continue
      
      # Draw the ommatidium
      draw3Dline(ommatidialPosition, ommatidialAxis, dwg);
  
  dwg.save()
      
if __name__ == "__main__":
  main(sys.argv[1:])
