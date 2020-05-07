// A header file to contain everything related to the Billboard primitive
#ifndef BILLBOARD_PRIMITIVE_H
#define BILLBOARD_PRIMITIVE_H

#include <optix.h> // For OptixAabb
#include <cuda_runtime.h> // For float vectors
#include <sutil/vec_math.h>// For vector maths
#include "SbtRecord.h" // for general SbtRecord definition template
#include "PrimitiveObject.h"

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

class BillboardPrimitive : public PrimitiveObject {
  public:
  BillboardData bbd;

  // Constructor and destructor
  BillboardPrimitive(float3 planeNormal, float3 planeOrigin, float radius, bool circular);
  ~BillboardPrimitive(void);

  // Override inherited methods
  inline int getNumberOfRequiredAttributeValues(){
    return 2;// Inline for speed
  };
  void recalculateProperties();

  // Unique methods
  private:
  OptixAabb bounds;
};

#endif
