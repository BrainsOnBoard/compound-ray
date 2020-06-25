#include "GenericCameraDataTypes.h"

struct PanoramicCameraData
{
  float startRadius = 0.0f;
};

// A typedef for a RaygenPosedContainer containing a PanoramicCameraData
typedef RaygenPosedContainer<PanoramicCameraData> PanoramicCameraPosedData;
