#include "PerspectiveCamera.h"

PerspectiveCamera::PerspectiveCamera(const std::string name) : DataRecordCamera<PerspectiveCameraData>(name)
{
  // Set the scale of the perspective camera
  sbtRecord.data.specializedData.scale = make_float3(100.0f, 100.0f, .1f);
}
PerspectiveCamera::~PerspectiveCamera()
{
}
