// The virtual superclass of all camera objects

#pragma once

#define DEBUG

#include <optix.h>
#include <sutil/Quaternion.h>
#include <sutil/Exception.h>

#include <iostream>

class Camera {
  public:
    // Define template Record type for SBT records:
    template <typename T>
    struct Record
    {
      __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
      T data;
    };

    // Define a typedef for a generic Empty Record
    struct EmptyData {};
    typedef Record<EmptyData> EmptyRecord;

    //Constructor/Destructor
    Camera();
    ~Camera();

    const float3& getPosition() const { return position; }
    void setPosition(const float3 pos);
    // Returns the local frame of the camera (always unit vectors)
    void getLocalFrame(float3& x, float3& y, float3& z) const;

    // Allocates device memory for the SBT record
    virtual void allocateRecord() = 0;
    // Packs and then copies the data onto the device
    virtual void packAndCopyRecord(OptixProgramGroup& programGroup) = 0;
    // Gets a pointer to the data on the device.
    const CUdeviceptr& getRecordPtr() const;

  protected:
    // The below allow access to device-side control objects
    CUdeviceptr d_record = 0;// Stores the SBT record required by this camera
    //const OptixProgramGroup& programGroup;// Stores a reference to the associated program group

  private:
    float3 position;
    //Quaternion orientation;
};
