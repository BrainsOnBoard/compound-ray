// A header file to contain the SBT data definitions for all common object shaders
#ifndef COMMON_OBJECT_SBT_DATAS_H
#define COMMON_OBJECT_SBT_DATAS_H

#include <optix>
#include "SbtRecord.h"

struct BillboardData
{
  float3 plane_normal;
  float3 plane_origin;
  float radius; // The radius of the billboard (maybe this should be 2d to draw rectangls/ovals?)
  bool circular; // Whether the billboard is circular or not
};

typedef SbtRecord<BillboardData> BillboardSbtRecord;
#endif
