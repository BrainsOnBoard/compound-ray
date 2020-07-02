#pragma once

#include "GenericCameraDataTypes.h"

// Define the perspective camera record
struct PerspectiveCameraData
{
  float3 scale;
  // x, y -> Aspect
  // z -> focal length/FOV

  inline bool operator==(const PerspectiveCameraData& other)
  { return (this->scale == other.scale); }
};

// A typedef for a RaygenPosedContainer containing a PerspectiveCameraData:
typedef RaygenPosedContainer<PerspectiveCameraData> PerspectiveCameraPosedData;
