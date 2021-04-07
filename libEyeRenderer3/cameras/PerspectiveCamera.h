#pragma once

#include <optix_stubs.h>// Needed for optixSbtRecordPackHeader

#include <sutil/sutilapi.h>
#include <sutil/vec_math.h>
#include "DataRecordCamera.h"
#include <sutil/Exception.h>

#include <iostream>

#include "PerspectiveCameraDataTypes.h"

class PerspectiveCamera : public DataRecordCamera<PerspectiveCameraData> {
  public:
    PerspectiveCamera(const std::string name);
    ~PerspectiveCamera();

    const char* getEntryFunctionName() const { return "__raygen__pinhole"; }

    // Sets the field of view (FOV) by taking the vertical FOV, in degrees.
    void setYFOV(float yFov);
    // Sets the field of view (FOV) by taking the horizontal FOV, in degrees.
    void setXFOV(float xFov);
    // Sets the field of view (FOV) by taking the diagonal FOV, from corner to corner, in degrees.
    //void setDiagonalFOV(float diagFov);
    // Sets the aspect ratio of the camera
    void setAspectRatio(float r);

  private:
    float aspectRatio = 1.0f;// Width to height
    float fromDegrees(float d);
};
