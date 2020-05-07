#include <optix.h>

#include "eyeRenderer.h"

#include <sutil/vec_math.h>


// Configuration variables
//rtDeclareVariable(float3, origin, , ); // The origin of this line
//rtDeclareVariable(float3, direction, , ); // The direction of this line (Must be normalised!)
//rtDeclareVariable(float, cylinderLength, , ); // The length of this line
//rtDeclareVariable(float, radius, , ); // The thickness of this line

// Recuired variables
//rtDeclareVariable(int, lgt_instance, , ) = {0};
//rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
//rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
//rtDeclareVariable(int, lgt_idx, attribute lgt_idx, ); 
//rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
}

static __forceinline__ __device__ float2 to2D(float3 v, float3 xAxis, float3 yAxis)
{
  return(make_float2(dot(v, xAxis), dot(v, yAxis)));
}
RT_PROGRAM void intersect(int)
{
  // Project the ray down to the 2D space with normal along the cylinder's axis (it's direction)
  // Perform circle-ray intersection between the ray and the cylinder's profile
  // Work out the scaling difference between the distance from the start of the ray to the intersection
  // That difference is the distance along the line.
  // scale the 3D ray by this distance to get the 3D intersection point
  // project to the 1D axis of the cylinder (it's direction)
  // If the projection is further than the cylinder's length, then no intersection occured.

  //// Calculate the intersection of the ray with the circle in 2D:

  // Create a 2D coordinate space that sits with the cylinder's direction as a normal.
  float3 xAxis;
  if(direction.x == 0.0f && direction.z == 0.0f) // Don't crossproduct a vertical cylinder
    xAxis = make_float3(1.0f, 0.0f, 0.0f);
  else
    xAxis = normalize(cross(direction, make_float3(0.0f,1.0f,0.0f)));
  float3 yAxis = normalize(cross(xAxis, direction));

  // Project the direction and origin (as in, the origin from the new coordinate system)
  float2 rayOrigin2D = to2D(ray.origin, xAxis, yAxis);
  float2 rayDir2D = to2D(ray.direction, xAxis, yAxis);
  float2 origin2D = to2D(origin, xAxis, yAxis);
  float2 cylinderToRay = origin2D - rayOrigin2D;
  float a = dot(rayDir2D, rayDir2D);
  if(a == 0.0f)
    return; // The ray does not intersect the cylinder.
  else
    a *= 2;// Multiply a by two to turn it into the denominator
  float b = -2*dot(rayDir2D, cylinderToRay); // Multiply by -1 so that it doesn't need to be later (disc uses b^2)
  float c = dot(cylinderToRay, cylinderToRay) - radius * radius;

  float disc = (b*b)-(2*a*c);

  if(disc < 0 || disc == 0.0f)
    return; // The ray does not intersect the cylinder or the ray intersects but only at the edges, which we don't care about.

  // The ray intersects the cylinder two times.
  // Now we can caluclate the two points on the circle. As we only care about rays coming
  // from the inside out, we can calculate each variation of the root and take the first
  // one that dots positively with the surface normal, meaning that it's viewing from
  // the inside.
  
  // Derive the correct non-backface intersect point:
  float sqrtDisc = sqrt(disc);
  float distance = (-b + sqrtDisc)/a;
  float2 intersectPoint2D = rayOrigin2D + distance * rayDir2D;
  float2 normal2D = normalize(origin2D - intersectPoint2D);
  float cullTest = dot((rayOrigin2D - intersectPoint2D), normal2D);
  if(cullTest > 0)// If the point is a back-face, then we'll have to calculate using the other point...
  {
    distance = (-b - sqrtDisc)/a;
    intersectPoint2D = rayOrigin2D + distance * rayDir2D;
    normal2D = normalize(origin2D - intersectPoint2D);
  }
  
  // Convert the 2D coordinates to 3D ones:
  float3 intersectPoint = ray.origin + distance * ray.direction;
  float3 normal = make_float3(normal2D.x, 0.0f, normal2D.y);// TODO: WRONG.
  //float3 normal = make_float3(dot(normal2D.x, make_float3(1.0f, 0.0f, 0.0f)), //TODO: This is right-er(?).
  //                            dot(direction, make_float3(0.0f, 1.0f, 0.0f)),
  //                            dot(normal2D.y, make_float3(0.0f, 0.0f, 1.0f))
  //                            );
  float distanceUpCylinder = dot((intersectPoint-origin), direction);
  //float distanceUpCylinder = dot((origin-intersectPoint), direction);

  // Render the object if the intersect point is within the accepted vertical space:
  if(distanceUpCylinder >= 0 && distanceUpCylinder <= cylinderLength && rtPotentialIntersection(distance))
  //if(rtPotentialIntersection(distance))
  {
    shading_normal = geometric_normal = normal;
    // Calculate the texture coordinates
    //textureCoords = make_float2((atan2f(intersectPoint2D.x-origin2D.x, intersectPoint2D.y-origin2D.y)+M_PIf)/M_PIf/2.0f, (intersectPoint.y - origin.y)/height);
    lgt_idx = lgt_instance;
    rtReportIntersection(0);// Render.
  }

}


RT_PROGRAM void bounds(int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  float3 end = origin + direction * cylinderLength;
  float3 tipRadius = make_float3(radius);
  aabb->set(fminf(origin, end) - tipRadius, fmaxf(origin, end) + tipRadius);
}
