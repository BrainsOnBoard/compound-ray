#include "PerspectiveCamera.h"


//void PerspectiveCamera::UVWFrame(float3& U, float3& V, float3& W) const
//{
//    float3 m_up = make_float3(0.0f, 1.0f, 0.0f);
//
//    //W = m_lookat - m_eye; // Do not normalize W -- it implies focal length
//    W = make_float3(1.0f, 0.0f, 0.0f);
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

PerspectiveCamera::PerspectiveCamera()
{
  // Allocate the SBT record for the associated raygen program
  allocateRecord();
  sbtRecord.data.scale = make_float3(100.0f, 100.0f, .1f);
}
PerspectiveCamera::~PerspectiveCamera()
{
}

void PerspectiveCamera::allocateRecord()
{
  #ifdef DEBUG
  std::cout << "Allocating Perspective camera SBT record on device." << std::endl;
  #endif
  if(d_record != 0)
  {
    #ifdef DEBUG
    std::cout << "WARN: Attempt to allocate perspective camera SBT record was made when one is already allocated." << std::endl;
    #endif
    return;
  }

  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_record ), recordSize) );
}

void PerspectiveCamera::packAndCopyRecord(OptixProgramGroup& programGroup)
{
  OPTIX_CHECK( optixSbtRecordPackHeader( programGroup, &sbtRecord) );
  CUDA_CHECK( cudaMemcpy(
              reinterpret_cast<void*>( d_record ),
              &sbtRecord,
              recordSize,
              cudaMemcpyHostToDevice
              ) );
}
