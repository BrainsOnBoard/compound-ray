#include "PrimitiveObject.h"

float3 PrimitiveObject::X_VECTOR = make_float3(1.0f, 0.0f, 0.0f);
float3 PrimitiveObject::Y_VECTOR = make_float3(0.0f, 1.0f, 0.0f);
float3 PrimitiveObject::Z_VECTOR = make_float3(0.0f, 0.0f, 1.0f);

PrimitiveObject::PrimitiveObject(void)
{
  dirty = true;
}
PrimitiveObject::~PrimitiveObject(void){}

void PrimitiveObject::recalculateIfDirty()
{
  if(dirty)
    recalculateProperties();
}

OptixAabb* PrimitiveObject::getBoundsPointer()
{
  return &bounds;
}

CUdeviceptr PrimitiveObject::allocateBoundsToDevice()
{
  CUdeviceptr d_bounds=0;
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_bounds ), sizeof(OptixAabb) ) );
  CUDA_CHECK( cudaMemcpy(
              reinterpret_cast<void*>( d_bounds ),
              getBoundsPointer(),
              sizeof(OptixAabb),
              cudaMemcpyHostToDevice
              ));
  return d_bounds;
}

void PrimitiveObject::appendIntersection(OptixProgramGroupDesc* opgd, OptixModule* mod)
{
  opgd->hitgroup.moduleIS = *mod;
  opgd->hitgroup.entryFunctionNameIS = "__intersection__intersect";
}

