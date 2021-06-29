// The virtual superclass of all camera objects

#pragma once

#include <optix.h>
#include <sutil/Quaternion.h>
#include <sutil/Exception.h>

#include <iostream>

//#include "GenericCameraDataTypes.h"

class GenericCamera {
  public:
    static constexpr const char* DEFAULT_RAYGEN_PROGRAM = "__raygen__pinhole";

    //Constructor/Destructor
    GenericCamera(const std::string name);
    virtual ~GenericCamera();

    virtual const float3 transformToLocal(const float3& vector) const = 0;
    virtual const float3& getPosition() const = 0;
    virtual void setPosition(const float3 pos) = 0;
    virtual void setLocalSpace(const float3 xAxis, const float3 yAxis, const float3 zAxis) = 0;
    virtual void lookAt(const float3& pos, const float3& upVector) = 0;
    // Rotates the camera 'angle' around the given axis
    virtual void rotateAround(const float angle, const float3& axis) = 0;
    // Rotates the camera 'angle' around the given axis, relative to the camera's local axis
    virtual void rotateLocallyAround(const float angle, const float3& axis) = 0;

    // Moves the camera by the given vector
    virtual void move(const float3& step) = 0;
    // Moves the camera by the given local vector
    virtual void moveLocally(const float3& localStep) = 0;

    // Packs and then copies the data onto the device (if the host-side representation has changed)
    virtual void packAndCopyRecordIfChanged(OptixProgramGroup& programGroup) = 0;
    // Forces the pack and copy of a record such that just-initialized cameras can be ensured to be memory-mapped
    virtual void forcePackAndCopyRecord(OptixProgramGroup& programGroup) = 0;
    // Gets a pointer to the data on the device.
    virtual const CUdeviceptr& getRecordPtr() const = 0;
    virtual const char* getEntryFunctionName() const { return DEFAULT_RAYGEN_PROGRAM; }

    const char* getCameraName() const { return camName.c_str(); }

    int32_t samplesPerPixel = 1;
  private:
    const std::string camName;
};
