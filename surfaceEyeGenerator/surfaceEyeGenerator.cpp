#include <iostream>
#include "Surface.h"

using namespace std;

void printHelp()
{
  cout << "USAGE:\nsurfaceEyeGenerator -f <path to surface .obj file>" << endl << endl;
  cout << "\t-h\tDisplay this help information." << endl;
}

int main(int argc, char** argv)
{
  cout << "Starting surface eye generator..." << endl;

  std::string path = "";

  // Parse inputs
  for(int i=0; i<argc; i++)
  {
    std::string arg = std::string(argv[i]);
    if(arg == "-h")
    {
      printHelp();
      return 0;
    }
    else if(arg == "-f")
    {
      i++;
      path = std::string(argv[i]);
    }
  }

  if(path == "")
  {
    printHelp();
    return 1;
  }

  Surface surface;
  if(!surface.load(path))
  {
    return 1;
  }

  surface.printVertexGraph();
  

  // Pre-calculate the surface's distances (also probably where the data that wraps the mesh is generated - such as neighbouring faces etc)
  // Open GL window? With tinyUI?
  // Place points
  // Iterate points to equilibrium
  // Save file
}


