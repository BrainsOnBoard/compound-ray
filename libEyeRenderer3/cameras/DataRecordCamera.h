#pragma once

//#define DEBUG

#include "GenericCameraDataTypes.h"
#include "GenericCamera.h"
#include <sutil/Quaternion.h>
#include <sutil/Matrix.h>

template<typename T>
class DataRecordCamera : public GenericCamera {
  public:
    DataRecordCamera(const std::string name) : GenericCamera(name)
    {
      // Allocate space for the record
      allocateRecord();
    }
    virtual ~DataRecordCamera()
    {
      // Free the allocated record
      freeRecord();
    }

    const float3& getPosition() const { return sbtRecord.data.position; }
    void setPosition(const float3 pos) 
    {
      sbtRecord.data.position.x = pos.x;
      sbtRecord.data.position.y = pos.y;
      sbtRecord.data.position.z = pos.z;
    }

    void setLocalSpace(const float3 xAxis, const float3 yAxis, const float3 zAxis)
    {
      ls.xAxis = xAxis;
      ls.yAxis = yAxis;
      ls.zAxis = zAxis;
    }
    void lookAt(const float3& pos)
    {
      lookAt(pos, make_float3(0.0f, 1.0f, 0.0f));
    }
    void lookAt(const float3& pos, const float3& upVector)
    {
      ls.zAxis = normalize(pos - sbtRecord.data.position);
      ls.xAxis = normalize(cross(ls.zAxis, upVector));
      ls.yAxis = normalize(cross(ls.xAxis, ls.zAxis));
    }
    void resetPose()
    {
      ls.xAxis = {1.0f, 0.0f, 0.0f};
      ls.yAxis = {0.0f, 1.0f, 0.0f};
      ls.zAxis = {0.0f, 0.0f, 1.0f};
      sbtRecord.data.position = {0.0f, 0.0f, 0.0f};
    }
      

    const float3 transformToLocal(const float3& vector) const
    {
      return (vector.x*ls.xAxis + vector.y*ls.yAxis + vector.z*ls.zAxis);
    }
    void rotateLocallyAround(const float angle, const float3& localAxis)
    {
      // Project the axis and then perform the rotation
      rotateAround(angle, transformToLocal(localAxis));
    }
    void rotateAround(const float angle, const float3& axis)
    {
      // Just performing an axis-angle rotation of the local space: A lot nicer.
      ls.xAxis = rotatePoint(ls.xAxis, angle, axis);
      ls.yAxis = rotatePoint(ls.yAxis, angle, axis);
      ls.zAxis = rotatePoint(ls.zAxis, angle, axis);
    }

    void moveLocally(const float3& localStep)
    {
      move(transformToLocal(localStep));
    }
    void move(const float3& step)
    {
      sbtRecord.data.position += step;
    }

    float3 rotatePoint(const float3& point, const float angle, const float3& axis)
    {
      const float3 normedAxis = normalize(axis);
      return (cos(angle)*point + sin(angle)*cross(normedAxis, point) + (1 - cos(angle))*dot(normedAxis, point)*normedAxis);
    }

    bool packAndCopyRecordIfChanged(OptixProgramGroup& programGroup)
    {
      // Only copy the data across if it's changed
      if(previous_sbtRecordData != sbtRecord.data)
      {
        #ifdef DEBUG
        std::cout << "ALERT: The following copy was triggered as the sbt record was flagged as changed:" <<std::endl;
        #endif
        forcePackAndCopyRecord(programGroup);
        return true;
      }
      return false;
    }

    void forcePackAndCopyRecord(OptixProgramGroup& programGroup)
    {
      #ifdef DEBUG
      std::cout<< "Copying device memory for camera '"<<getCameraName()<<"'."<<std::endl;
      printf("\td_record: %p\n", d_record);
      std::cout<< "\tproggroup: "<< programGroup<<std::endl;
      std::cout<< "\tPosition: ("<<sbtRecord.data.position.x<<", "
                                 <<sbtRecord.data.position.y<<", "
                                 <<sbtRecord.data.position.z<<")"<<std::endl;
      std::cout<< "\tLocalSpace: (("<<sbtRecord.data.localSpace.xAxis.x<<", "
                                    <<sbtRecord.data.localSpace.xAxis.y<<", "
                                    <<sbtRecord.data.localSpace.xAxis.z<<")"<<std::endl << "\t              "
                                    <<sbtRecord.data.localSpace.yAxis.x<<", "
                                    <<sbtRecord.data.localSpace.yAxis.y<<", "
                                  <<sbtRecord.data.localSpace.yAxis.z<<")"<<std::endl << "\t              "
                                  <<sbtRecord.data.localSpace.zAxis.x<<", "
                                  <<sbtRecord.data.localSpace.zAxis.y<<", "
                                  <<sbtRecord.data.localSpace.zAxis.z<<")"<<std::endl;
      #endif
      OPTIX_CHECK( optixSbtRecordPackHeader( programGroup, &sbtRecord) );
      CUDA_CHECK( cudaMemcpy(
                  reinterpret_cast<void*>( d_record ),
                  &sbtRecord,
                  sizeof(sbtRecord),
                  cudaMemcpyHostToDevice
                  ) );
      previous_sbtRecordData = sbtRecord.data;
    }

    virtual const CUdeviceptr& getRecordPtr() const {return d_record;}

  protected:
    //RaygenPosedContainerRecord<T> sbtRecord; // The below is also of this type
    RaygenRecord<RaygenPosedContainer<T>> sbtRecord; // The sbtRecord associated with this camera
    T& specializedData = sbtRecord.data.specializedData; // Convenience reference
    LocalSpace& ls = sbtRecord.data.localSpace; // Convenience reference

  private:
    static const LocalSpace BASE_LOCALSPACE;// A base localspace to use for rotations.

    CUdeviceptr d_record = 0;// Stores the pointer to the SBT record 

    // Change tracking duplicates (done by keeping an old copy and comparing)
    RaygenPosedContainer<T> previous_sbtRecordData;

    void allocateRecord()
    {
      #ifdef DEBUG
      std::cout << "Allocating camera SBT record on device (size: "<< sizeof(sbtRecord) << ")..." << std::endl;
      #endif
      if(d_record != 0)
      {
        #ifdef DEBUG
        std::cout << "  WARN: Attempt to allocate camera SBT record was made when one is already allocated." << std::endl;
        #endif
        return;
      }

      CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_record ), sizeof(sbtRecord)) );
      #ifdef DEBUG
      printf("  ...allocated at %p\n", d_record);
      #endif
    }
    void freeRecord()
    {
      #ifdef DEBUG
      std::cout << "Freeing camera SBT record..." << std::endl;
      #endif
      if(d_record != 0)
      {
        CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_record)) );
      }
    }
};
