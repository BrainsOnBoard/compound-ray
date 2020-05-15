// A header file to contain everything related to the Billboard primitive
#ifndef BILLBOARD_PRIMITIVE_H
#define BILLBOARD_PRIMITIVE_H

#include <optix.h> // For OptixAabb
#include <cuda_runtime.h> // For float vectors
#include <sutil/vec_math.h>// For vector maths
#include <sutil/sutil.h>
#include "PrimitiveObject.h"
#include "BillboardData.h"


class BillboardPrimitive : public PrimitiveObject {
  public:

    BillboardData bbd;

    // Constructor and destructor
    BillboardPrimitive(float3 planeNormal, float3 planeOrigin, float radius, bool circular);
    ~BillboardPrimitive(void);

    // Override inherited methods
    inline int getNumberOfRequiredAttributeValues(){
      return 5;// 2 for the UV coordinates, 3 for the normal
    };
    void recalculateProperties();
    OptixModule createOptixModule(OptixPipelineCompileOptions pipelineCompileOptions, OptixDeviceContext* contextPtr, char* log, size_t sizeof_log);

    // Unique methods
  private:
    // Nada
};

#endif
