#include "PerspectiveCamera.h"

PerspectiveCamera::PerspectiveCamera(const std::string name) : DataRecordCamera<PerspectiveCameraData>(name)
{
  // Set the scale of the perspective camera
  specializedData.scale = make_float3(10.0f, 10.0f, 1.0f);
}
PerspectiveCamera::~PerspectiveCamera()
{
}

void PerspectiveCamera::setXFOV(float xFOV)
{
  xFOV = fromDegrees(xFOV);
  specializedData.scale.x = tan(xFOV/2.0f) * specializedData.scale.z;
  specializedData.scale.y = specializedData.scale.y/aspectRatio;
}
void PerspectiveCamera::setYFOV(float yFOV)
{
  yFOV = fromDegrees(yFOV);
  specializedData.scale.y = tan(yFOV/2.0f) * specializedData.scale.z;
  specializedData.scale.x = specializedData.scale.y * aspectRatio;
}
void PerspectiveCamera::setAspectRatio(float r)
{
  aspectRatio = r;
  float previousYfov = atan(specializedData.scale.y/specializedData.scale.z)*2.0f;
  setYFOV(previousYfov);
}

inline float PerspectiveCamera::fromDegrees(float d)
{
  return (d/180 * M_PIf);
}
