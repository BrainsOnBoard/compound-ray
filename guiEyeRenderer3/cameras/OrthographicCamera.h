#pragma once

#include "GenericCamera.h"
#include "DataRecordCamera.h"

#include <optix_stubs.h>// Needed for optixSbtRecordPackHeader

#include "OrthographicCameraDataTypes.h"

class OrthographicCamera : public DataRecordCamera<OrthographicCameraData> {
  public:
    OrthographicCamera(const std::string name);
    ~OrthographicCamera();

    const char* getEntryFunctionName() const { return "__raygen__orthographic"; }

    void setXYscale(float x, float y);
    void setXYscale(float2 scale) {setXYscale(scale.x, scale.y);}

  private:
};
