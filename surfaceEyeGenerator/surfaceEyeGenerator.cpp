// Include tinyObjLoader
//#define TINYOBJLOADER_USE_DOUBLE // Enable double-precision object loading
#define TINYOBJLOADER_IMPLEMENTATION
#include <support/tinyobjloader/tiny_obj_loader.h>

#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
  cout << "Starting surface eye generator..." << endl;

  // Load the surface using tiny OBJ loader ( https://github.com/tinyobjloader/tinyobjloader )
  cout << "Loading surface..." << endl;
  

  // Pre-calculate the surface's distances (also probably where the data that wraps the mesh is generated - such as neighbouring faces etc)
  // Open GL window? With tinyUI?
  // Place points
  // Iterate points to equilibrium
  // Save file
}
