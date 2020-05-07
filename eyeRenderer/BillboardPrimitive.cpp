#include "BillboardPrimitive.h"

// Constructor
BillboardPrimitive::BillboardPrimitive(float3 planeNormal, float3 planeOrigin, float radius, bool circular)
{
  bbd.planeNormal = planeNormal;
  bbd.planeOrigin = planeOrigin;
  bbd.radius = radius;
  bbd.circular = circular;
  recalculateProperties();
}
// Destructor
BillboardPrimitive::~BillboardPrimitive(void)
{
  
}

void BillboardPrimitive::recalculateProperties()
{
  //// Calculate BillboardData axes
  if(bbd.planeNormal.x == Y_VECTOR.x && bbd.planeNormal.y == Y_VECTOR.y && bbd.planeNormal.z == Y_VECTOR.z)
  //if(bbd.planeNormal == Y_VECTOR)
    bbd.precalc_xAxis = X_VECTOR;// If the normal is also pointed up, solve gimble lock
  else
    bbd.precalc_xAxis = normalize(cross(Y_VECTOR, bbd.planeNormal));// If not, form the xAxis
  bbd.precalc_yAxis = normalize(cross(bbd.planeNormal, bbd.precalc_xAxis));

  //// Calculate the bounding volume
  // Calculate the position of each of it's corners, then use it to create the Aabb
  float3 v1 = bbd.radius * ( bbd.precalc_xAxis + bbd.precalc_yAxis);
  float3 v2 = bbd.radius * ( bbd.precalc_xAxis - bbd.precalc_yAxis);
  float3 v3 = bbd.radius * (-bbd.precalc_xAxis - bbd.precalc_yAxis);
  float3 v4 = bbd.radius * (-bbd.precalc_xAxis + bbd.precalc_yAxis);
  float3 vMin = bbd.planeOrigin + fminf(fminf(v1,v2),fminf(v3,v4));
  float3 vMax = bbd.planeOrigin + fmaxf(fmaxf(v1,v2),fmaxf(v3,v4));
  setBounds(vMin, vMax);
}
