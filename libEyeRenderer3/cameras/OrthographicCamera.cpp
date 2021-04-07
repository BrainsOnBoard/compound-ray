#include "OrthographicCamera.h"

OrthographicCamera::OrthographicCamera(const std::string name) : DataRecordCamera<OrthographicCameraData>(name)
{
}
OrthographicCamera::~OrthographicCamera() {}

void OrthographicCamera::setXYscale(float x, float y)
{
  specializedData.scale.x = x;
  specializedData.scale.y = y;
}
