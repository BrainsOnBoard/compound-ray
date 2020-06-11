#pragma once

#include <optix_stubs.h>// Needed for optixSbtRecordPackHeader

#include <sutil/sutilapi.h>
#include <sutil/vec_math.h>
#include "Camera.h"
#include <sutil/Exception.h>

#include <iostream>

class PerspectiveCamera : public Camera {
  public:
    // Define the perspective camera record
    struct PerspectiveCameraData
    {
      float3 scale;
      // x, y -> Aspect
      // z -> focal length/FOV
    };
    typedef Record<PerspectiveCameraData> PerspectiveCameraRecord;

    PerspectiveCamera();
    ~PerspectiveCamera();

    void allocateRecord();
    void packAndCopyRecord(OptixProgramGroup& programGroup);

//    // Sets the field of view (FOV) by taking the vertical FOV, in degrees.
//    void setYFOV(float yFov);
//    // Sets the field of view (FOV) by taking the horizontal FOV, in degrees.
//    void setXFOV(float xFov);
//    // Sets the field of view (FOV) by taking the diagonal FOV, from corner to corner, in degrees.
//    void setDiagonalFOV(float diagFov);
//    // Sets the aspect ratio of the camera
//    void setAspectRatio(float r);

  private:
    float aspectRatio = 1.0f;
    PerspectiveCameraRecord sbtRecord;
    const size_t recordSize = sizeof(PerspectiveCameraRecord);
};
