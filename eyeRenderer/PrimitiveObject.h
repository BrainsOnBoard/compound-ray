#ifndef PRIMITIVE_OBJECT_H
#define PRIMITIVE_OBJECT_H

#include <cuda_runtime.h>
#include <optix.h>
#include <iostream>
#include <optix_stubs.h>
#include <sutil/Exception.h>

class PrimitiveObject {
  public:
    static float3 X_VECTOR;
    static float3 Y_VECTOR;
    static float3 Z_VECTOR;

    PrimitiveObject();
    virtual ~PrimitiveObject();

    // Returns the number of required attribute values
    virtual int getNumberOfRequiredAttributeValues() = 0;
    // Updates the properties of this primitive
    virtual void recalculateProperties() = 0;

    // Gets a reference to the bounding box for this primitive
    OptixAabb* getBoundsPointer();
    // Recalculates the properties of this object
    void recalculateIfDirty();
    // Allocates the bounding box of the primitive object to device memory
    CUdeviceptr allocateBoundsToDevice();

  protected:
    OptixAabb bounds;
    inline void setBounds(float3 vMin, float3 vMax)
    {
      // Inline for speed
      bounds.minX = vMin.x;
      bounds.minY = vMin.y;
      bounds.minZ = vMin.z;
      bounds.maxX = vMax.x;
      bounds.maxY = vMax.y;
      bounds.maxZ = vMax.z;
    }

  private:
    bool dirty;
};

#endif
