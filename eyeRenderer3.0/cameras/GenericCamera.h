// The virtual superclass of all camera objects

#pragma once

#define DEBUG

#include <optix.h>
#include <sutil/Quaternion.h>
#include <sutil/Exception.h>

#include <iostream>

//#include "GenericCameraDataTypes.h"

class GenericCamera {
  public:
    static constexpr char* DEFAULT_RAYGEN_PROGRAM = "__raygen__pinhole";

    //Constructor/Destructor
    GenericCamera(const std::string name);
    virtual ~GenericCamera();

    virtual const float3& getPosition() const = 0;
    virtual void setPosition(const float3 pos) = 0;
    // Returns the local frame of the camera (always unit vectors)
    virtual void getLocalFrame(float3& x, float3& y, float3& z) const = 0;

    // Rotates the camera 'angle' around the given axis
    virtual void rotateAround(const float angle, const float3& axis) = 0;

    // Packs and then copies the data onto the device (if the host-side representation has changed)
    virtual void packAndCopyRecordIfChanged(OptixProgramGroup& programGroup) = 0;
    // Forces the pack and copy of a record such that just-initialized cameras can be ensured to be memory-mapped
    virtual void forcePackAndCopyRecord(OptixProgramGroup& programGroup) = 0;
    // Gets a pointer to the data on the device.
    virtual const CUdeviceptr& getRecordPtr() const = 0;
    virtual const char* getEntryFunctionName() const { return DEFAULT_RAYGEN_PROGRAM; }

    virtual void UVWFrame(float3& U, float3& V, float3& W) const = 0;

    const char* getCameraName() const { return camName.c_str(); }


  private:
    const std::string camName;
};
