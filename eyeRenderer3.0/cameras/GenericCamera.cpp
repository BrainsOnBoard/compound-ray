
#include "GenericCamera.h"

//void GenericCamera::UVWFrame(float3& U, float3& V, float3& W) const
//{
//    float3 m_up = make_float3(0.0f, 1.0f, 0.0f);
//
//    float3 m_lookat = make_float3(0.0f);
//    W = m_lookat - position; // Do not normalize W -- it implies focal length
//    //W = make_float3(1.0f, 0.0f, 0.0f);
//    float wlen = length(W);
//    U = normalize(cross(W, m_up));
//    V = normalize(cross(U, W));
//
//    float m_fovY = 120.0f;
//    float m_aspectRatio = 1.0f;
//
//    float vlen = wlen * tanf(0.5f * m_fovY * M_PIf / 180.0f);
//    V *= vlen;
//    float ulen = vlen * m_aspectRatio;
//    U *= ulen;
//}

GenericCamera::GenericCamera() : position(make_float3(0.0f))
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
  std::cout<<"Running Generic-side get local frame." << std::endl;
  //x = make_float3(1.0f, 0.0f, 0.0f);
  //y = make_float3(0.0f, 1.0f, 0.0f);
  //z = make_float3(0.0f, 0.0f, 1.0f);

  float3 m_up = make_float3(0.0f, 1.0f, 0.0f);

  float3 m_lookat = make_float3(0.0f);
  //z = normalize(m_lookat - position); // Do not normalize W -- it implies focal length
  z = m_lookat - position; // Do not normalize W -- it implies focal length
  float wlen = length(z);
  x = normalize(cross(z, m_up));
  y = normalize(cross(x, z));

  float m_fovY = 120.0f;
  float m_aspectRatio = 1.0f;

  float vlen = wlen * tanf(0.5f * m_fovY * M_PIf / 180.0f);
  y *= vlen;
  float ulen = vlen * m_aspectRatio;
  x *= ulen;
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

