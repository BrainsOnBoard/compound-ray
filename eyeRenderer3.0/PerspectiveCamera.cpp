#include "PerspectiveCamera.h"



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
