# Orthogonally renders a given eye to an SVG file
# Uses svgwrite: https://github.com/mozman/svgwrite

import sys
import svgwrite
import numpy as np
import getopt

def helpAndExit(error):
	USAGE_STRING = """
USAGE:
python eyeVisualiser.py -f <eye file path> 

Optionals:
-y <yaw>\thorizontal yaw of the camera (azimuth)
-p <pitch>\tvertical pitch of the camera (altitude)
-w <width>\twidth of the image> -h <height of the image
-o <outpath>\toutput image path (svg, png)
-b \t\tSwitch to enable rendering of away-facing ommatidia"""
	print("Error:", error)
	print(USAGE_STRING)
	sys.exit()

def main(argv):
	# Defaults
	infilePath = None
	yaw = 0
	pitch = 0
	imgWidth = 100
	imgHeight = 100
	outputPath = "visualisedEye.svg"
	renderBackfaces = False
	
	# Get the input parameters
	try:
		opts, args = getopt.getopt(argv, "f:y:p:w:h:o:b")
	except getopt.GetoptError:
		helpAndExit()
	
	for opt, arg in opts:
		if opt == "--help":
			helpAndExit()
		elif opt == "-f":
			infilePath = arg
		elif opt == "-y":
			yaw = float(arg)
		elif opt == "-p":
			pitch = float(arg)
		elif opt == "-w":
			imgWidth = int(arg)
		elif opt == "-h":
			imgHeight = int(arg)
		elif opt == "-o":
			outputPath = arg
		elif opt == "-b":
			renderBackfaces = True
		
	if infilePath == None:
		helpAndExit("Please provide an input eye filepath.")
		
	# Read and render the eye
	with open(infilePath, "r") as eyeFile:
		for line in eyeFile:
			print(line)
			
if __name__ == "__main__":
	main(sys.argv[1:])