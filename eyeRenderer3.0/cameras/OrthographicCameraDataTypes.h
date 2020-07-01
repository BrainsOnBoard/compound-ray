#include "GenericCameraDataTypes.h"

struct OrthographicCameraData
{
  float2 scale;
};

// A typedef for a RaygenPosedContainer containing an OrthographicCameraData
typedef RaygenPosedContainer<OrthographicCameraData> OrthographicCameraPosedData;
