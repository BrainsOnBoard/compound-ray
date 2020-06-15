#pragma once

#include <optix_stubs.h>// Needed for optixSbtRecordPackHeader

#include <sutil/sutilapi.h>
#include <sutil/vec_math.h>
#include "GenericCamera.h"
#include <sutil/Exception.h>

#include <iostream>

#include "PerspectiveCameraDataTypes.h"

class PerspectiveCamera : public GenericCamera {
  public:
    static const int PROGRAM_GROUP_ID = 1;// Horrible hacky work around

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
