#include <optix.h>

#include <sutil/vec_math.h>
#include "CommonObjectSbtDatas.h"

RT_PROGRAM void intersect(int)
{
  //if(threadIdx.x == 10)
  //  printf("Hello from new floor.\n");

  //// Calculate the distance to the surface
  float distance = -dot(ray.origin - plane_origin, plane_normal)/dot(ray.direction, plane_normal);

  //// Calculate the UV coordinates of the intersection point on the surface
  // Calculate the intersection
  float3 intersectionPoint = ray.origin + (ray.direction * distance);
  // Now we have the coordinate-space vectors of the plane, they can be scaled to the appropriate size

  // Now dotted with the vector forming from the centre to the intersection point (the 0-1 scaled UV coordinate):
  float3 localVector = intersectionPoint - plane_origin;

  //// If it's circular, we only need to know the distance along the plane from the origin to the intersect point
  if(circular == 1)// Default is false.
  {
    if(length(localVector) <= radius && rtPotentialIntersection(distance)) // If it's inside the radius of the circle and visible
    {
        shading_normal = geometric_normal = plane_normal;
        lgt_idx = lgt_instance;
        rtReportIntersection(0);// Render
    }else{
      return;// If not, then don't.
    }
  }

  //// Place into UVspace one axis at a time, checkinf if it's within boounds at each:
  // The plane is `radius*2` x `radius*2` large, but it's UV coordinates run from -1 to 1.
  float u = dot(precalc_xAxis, localVector);
  if(u >= -radius && u <= radius)
  {
    float v = dot(precalc_yAxis, localVector);
    if(v >= -radius && v <= radius)
    {
      //// Return the intersection if within the bounds of the shape.
      if(rtPotentialIntersection(distance))
      {
        shading_normal = geometric_normal = plane_normal;
        uv_coords = (make_float2(u,v) - make_float2(radius,radius))/(2*radius);
        lgt_idx = lgt_instance;
        rtReportIntersection(0);
      }
    }
  }
}


RT_PROGRAM void bounds (int, float result[6])
{
  //printf("Hello from new billboard bounds.\n");
  //// Calculate the maximum bounds this object has by calculating the positions of each of it's corners.
  float3 v1 = radius * (precalc_xAxis + precalc_yAxis);
  float3 v2 = radius * (precalc_xAxis - precalc_yAxis);
  float3 v3 = radius * (-precalc_xAxis - precalc_yAxis);
  float3 v4 = radius * (-precalc_xAxis + precalc_yAxis);
  float3 vMin = plane_origin + fminf(fminf(v1,v2),fminf(v3,v4));
  float3 vMax = plane_origin + fmaxf(fmaxf(v1,v2),fmaxf(v3,v4));
  //printf("  | vMin: (%f, %f, %f)\n", vMin.x, vMin.y, vMin.z);
  //printf("  | vMax: (%f, %f, %f)\n", vMax.x, vMax.y, vMax.z);
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->set(vMin, vMax);
}
