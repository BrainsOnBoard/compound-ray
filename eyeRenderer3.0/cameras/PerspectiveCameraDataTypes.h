#pragma once

#include "GenericCameraDataTypes.h"

// Define the perspective camera record
struct PerspectiveCameraData
{
  float3 scale;
  // x, y -> Aspect
  // z -> focal length/FOV
};
typedef RaygenRecord<PerspectiveCameraData> PerspectiveCameraRecord;
