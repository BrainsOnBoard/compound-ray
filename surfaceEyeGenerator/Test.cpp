#include <iostream>
#include "NonPlanarCoordinate.h"

//#include <cuda_runtime.h>
//#include <cuda.h>
#include <sutil/vec_math.h>

int main( int argc, char* argv[])
{
  std::cout << "THIS IS A TEST!" << std::endl;
  float3 thisIsAFloat = make_float3(1.0f, 2.0f, 3.0f);
  std::cout << "Float: (" << thisIsAFloat.x << ", " << thisIsAFloat.y << ", " << thisIsAFloat.z << ")" << std::endl;
  thisIsAFloat = thisIsAFloat * 2.0f;
  std::cout << "Float: (" << thisIsAFloat.x << ", " << thisIsAFloat.y << ", " << thisIsAFloat.z << ")" << std::endl;
}
