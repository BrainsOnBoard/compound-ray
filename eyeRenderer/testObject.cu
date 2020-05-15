#include <optix.h>

//#include <cuda.h>
//#include <curand_kernel.h>

#include <sutil/vec_math.h>
#include "TestObjectSbt.h"


extern "C" __global__ void __closesthit__closehit()
{

  //if(threadIdx.x == 10)
  //  printf("Hello from the closehit call.\n");
  // To get the attributes:
  // optixGetAttribute_0() (remember ot recast using __int_as_float from __float_as_int)
  //setPayload( make_float3( 1.0f, 1.0f , 1.0f ) );
  //optixGetSbtDataPointer());
  const TestObjectData* toData = (TestObjectData*)optixGetSbtDataPointer();
  optixSetPayload_0(float_as_int(toData->r));
  optixSetPayload_1(float_as_int(toData->g));
  optixSetPayload_2(float_as_int(toData->b));
}
extern "C" __global__ void __intersection__intersect()
{
  //if(threadIdx.x == 10)
  //  printf("Hello from the new intersection call.\n");
  optixReportIntersection(2.0f, 0);
  //optixGetWorldRayDirection(); // <-- this gets the... well, uh.. you get the point.
  //optixGetWorldRayOrigin();
}
