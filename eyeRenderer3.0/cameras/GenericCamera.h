// The virtual superclass of all camera objects

#pragma once

#define DEBUG

#include <optix.h>
#include <sutil/Quaternion.h>
#include <sutil/Exception.h>

#include <iostream>

#include "GenericCameraDataTypes.h"

class GenericCamera {
  public:
    static constexpr char* DEFAULT_RAYGEN_PROGRAM = "__raygen__pinhole";

    //Constructor/Destructor
    GenericCamera();
    ~GenericCamera();

    const float3& getPosition() const { return position; }
    void setPosition(const float3 pos);
    // Returns the local frame of the camera (always unit vectors)
    virtual void getLocalFrame(float3& x, float3& y, float3& z) const;

    // Allocates device memory for the SBT record
    virtual void allocateRecord() = 0;
    // Packs and then copies the data onto the device
    virtual void packAndCopyRecord(OptixProgramGroup& programGroup) = 0;
    // Gets a pointer to the data on the device.
    const CUdeviceptr& getRecordPtr() const;
    virtual const char* getEntryFunctionName() const { return DEFAULT_RAYGEN_PROGRAM; }

    void UVWFrame(float3& U, float3& V, float3& W) const;

  protected:
    // The below allow access to device-side control objects
    CUdeviceptr d_record = 0;// Stores the SBT record required by this camera
    //const OptixProgramGroup& programGroup;// Stores a reference to the associated program group

  private:
    float3 position;
    //Quaternion orientation;
};
