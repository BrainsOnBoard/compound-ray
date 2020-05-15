#ifndef BILLBOARD_DATA_H
#define BILLBOARD_DATA_H
#include "SbtRecord.h" // for general SbtRecord definition template

struct BillboardData
{
  float3 planeNormal;
  float3 planeOrigin;
  float radius; // The radius of the billboard (maybe this should be 2d to draw rectangls/ovals?)
  bool circular; // Whether the billboard is circular or not

  // Precalculated coordinate-space vectors of the plane
  float3 precalc_xAxis;
  float3 precalc_yAxis;
};

typedef SbtRecord<BillboardData> BillboardSbtRecord;
#endif
