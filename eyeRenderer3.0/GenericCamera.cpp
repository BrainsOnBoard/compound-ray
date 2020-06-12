
#include "GenericCamera.h"

GenericCamera::GenericCamera() : position(make_float3(0.0f)), programGroupID(0)
{
  #ifdef DEBUG
  std::cout << "Creating camera..." << std::endl;
  #endif
}
GenericCamera::GenericCamera(int progGroupID) : position(make_float3(0.0f)), programGroupID(progGroupID)
{
  #ifdef DEBUG
  std::cout << "Creating camera..." << std::endl;
  #endif
}
GenericCamera::~GenericCamera()
{
  if(d_record !=  0)
  {
    #ifdef DEBUG
    std::cout << "Freeing camera SBT record on device." << std::endl;
    #endif
    //CUDA_CHECK( cudaFree((void*)d_record) );
    //d_record = 0;
  }
}

void GenericCamera::getLocalFrame(float3& x, float3& y, float3& z) const
{
  x = make_float3(1.0f, 0.0f, 0.0f);
  y = make_float3(0.0f, 1.0f, 0.0f);
  z = make_float3(0.0f, 0.0f, 1.0f);
}

void GenericCamera::setPosition(const float3 pos)
{
  position.x = pos.x;
  position.y = pos.y;
  position.z = pos.z;
}

const CUdeviceptr& GenericCamera::getRecordPtr() const
{
  return d_record;
}

