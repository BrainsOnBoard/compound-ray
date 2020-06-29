#include "PerspectiveCamera.h"

PerspectiveCamera::PerspectiveCamera(const std::string name) : DataRecordCamera<PerspectiveCameraData>(name)
{
  // Set the scale of the perspective camera
  sbtRecord.data.specializedData.scale = make_float3(10.0f, 10.0f, 1.0f);
}
PerspectiveCamera::~PerspectiveCamera()
{
}

void PerspectiveCamera::setXFOV(float xFOV)
{
  xFOV = fromDegrees(xFOV);
  perspectiveData.scale.x = tan(xFOV/2.0f) * perspectiveData.scale.z;
  perspectiveData.scale.y = perspectiveData.scale.y/aspectRatio;
}
void PerspectiveCamera::setYFOV(float yFOV)
{
  yFOV = fromDegrees(yFOV);
  perspectiveData.scale.y = tan(yFOV/2.0f) * perspectiveData.scale.z;
  perspectiveData.scale.x = perspectiveData.scale.y * aspectRatio;
}
void PerspectiveCamera::setAspectRatio(float r)
{
  aspectRatio = r;
  float previousYfov = atan(perspectiveData.scale.y/perspectiveData.scale.z)*2.0f;
  setYFOV(previousYfov);
}

inline float PerspectiveCamera::fromDegrees(float d)
{
  return (d/180 * M_PIf);
}
