
#include "Camera.h"

Camera::Camera() : position(make_float3(0.0f))
{
  #ifdef DEBUG
  std::cout << "Creating camera..." << std::endl;
  #endif
}
Camera::~Camera()
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

void Camera::getLocalFrame(float3& x, float3& y, float3& z) const
{
  x = make_float3(1.0f, 0.0f, 0.0f);
  y = make_float3(0.0f, 1.0f, 0.0f);
  z = make_float3(0.0f, 0.0f, 1.0f);
}

void Camera::setPosition(const float3 pos)
{
  position.x = pos.x;
  position.y = pos.y;
  position.z = pos.z;
}

const CUdeviceptr& Camera::getRecordPtr() const
{
  return d_record;
}
