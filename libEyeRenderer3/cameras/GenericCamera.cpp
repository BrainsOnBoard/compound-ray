
#include "GenericCamera.h"


GenericCamera::GenericCamera(const std::string name) : camName(name)
{
  #ifdef DEBUG
  std::cout << "Creating camera..." << std::endl;
  #endif
}
GenericCamera::~GenericCamera()
{
}
