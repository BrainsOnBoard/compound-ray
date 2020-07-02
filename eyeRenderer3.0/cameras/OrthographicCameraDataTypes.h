#include "GenericCameraDataTypes.h"

struct OrthographicCameraData
{
  float2 scale;

  inline bool operator==(const OrthographicCameraData& other)
  { return (this->scale == other.scale); }
};

// A typedef for a RaygenPosedContainer containing an OrthographicCameraData
typedef RaygenPosedContainer<OrthographicCameraData> OrthographicCameraPosedData;
