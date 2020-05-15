#include <optix.h>

#include <sutil/vec_math.h>
#include "BillboardData.h"

extern "C" __global__ void __intersection__intersect()
{
  //optixReportIntersection(2.0f,0);
  const float3 rayDirection = optixGetWorldRayDirection();
  const float3 rayOrigin = optixGetWorldRayOrigin();
  BillboardData* bbd = (BillboardData*)optixGetSbtDataPointer();

  //// Calculate the distance to the surface
  float distance = -dot(rayOrigin - bbd->planeOrigin, bbd->planeNormal)/dot(rayDirection, bbd->planeNormal);

  //// Calculate the UV coordinates of the intersection point on the surface
  // Calculate the intersection
  float3 intersectionPoint = rayOrigin + (rayDirection * distance);
  // Now we have the coordinate-space vectors of the plane, they can be scaled to the appropriate size

  // Now dotted with the vector forming from the centre to the intersection point (the 0-1 scaled UV coordinate):
  float3 localVector = intersectionPoint - bbd->planeOrigin;

  //// If it's circular, we only need to know the distance along the plane from the origin to the intersect point
    if(threadIdx.x == 10)
    {
      printf("%f\n", bbd->radius);
    }
  if(bbd->circular == true)
  {
    if(threadIdx.x == 10)
    {
      printf("got here\n");
      //printf("%f\n", distance);
    }
    if(length(localVector) <= bbd->radius) // If it's inside the radius of the circle
    {
        optixReportIntersection(distance, 1);
    }else{
      return;// If not, then don't.
    }
  }

  ////// Place into UVspace one axis at a time, checkinf if it's within boounds at each:
  //// The plane is `(bbd->radius)*2` x `(bbd->radius)*2` large, but it's UV coordinates run from -1 to 1.
  //float u = dot(bbd->precalc_xAxis, localVector);
  //if(u >= -(bbd->radius) && u <= bbd->radius)
  //{
  //  float v = dot(bbd->precalc_yAxis, localVector);
  //  if(v >= -(bbd->radius) && v <= bbd->radius)
  //  {
  //    //// Return the intersection if within the bounds of the shape.
  //    const float2 uv_coords = (make_float2(u,v) - make_float2((bbd->radius),(bbd->radius)))/(2*(bbd->radius));
  //    optixReportIntersection(distance, 1,
  //      float_as_int(bbd->planeNormal.x),
  //      float_as_int(bbd->planeNormal.y),
  //      float_as_int(bbd->planeNormal.z),
  //      float_as_int(uv_coords.x),
  //      float_as_int(uv_coords.y)
  //    );
  //  }
  //}
}
extern "C" __global__ void __closesthit__closehit()
{
  optixSetPayload_0(float_as_int(0.0f));
  optixSetPayload_1(float_as_int(1.0f));
  optixSetPayload_2(float_as_int(0.0f));
}
