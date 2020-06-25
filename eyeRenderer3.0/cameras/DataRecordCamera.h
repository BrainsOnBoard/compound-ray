#pragma once

#define DEBUG

#include "GenericCameraDataTypes.h"
#include "GenericCamera.h"
#include <sutil/Quaternion.h>

template<typename T>
class DataRecordCamera : public GenericCamera {
  public:
    DataRecordCamera(const std::string name) : GenericCamera(name)
    {
      std::cout<<"RUNNING DATA RECORD CAM CREATION."<<std::endl;
      // Allocate space for the record
      allocateRecord();
    }
    virtual ~DataRecordCamera()
    {
      std::cout<<"RUNNING DATA RECORD CAM DESTRUCTION."<<std::endl;
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

    //const Quaternion

    void getLocalFrame(float3& x, float3& y, float3& z) const
    {
      std::cout<<"Running DataRecordCamera-side get local frame." << std::endl;
      float3 m_up = make_float3(0.0f, 1.0f, 0.0f);

      float3 m_lookat = make_float3(0.0f);
      //z = normalize(m_lookat - sbtRecord.position); // Do not normalize W -- it implies focal length
      z = m_lookat - sbtRecord.data.position; // Do not normalize W -- it implies focal length
      float wlen = length(z);
      x = normalize(cross(z, m_up));
      y = normalize(cross(x, z));

      float m_fovY = 120.0f;
      float m_aspectRatio = 1.0f;

      float vlen = wlen * tanf(0.5f * m_fovY * M_PIf / 180.0f);
      y *= vlen;
      float ulen = vlen * m_aspectRatio;
      x *= ulen;
    }

    void UVWFrame(float3& U, float3& V, float3& W) const
    {
        float3 m_up = make_float3(0.0f, 1.0f, 0.0f);

        float3 m_lookat = make_float3(0.0f);
        W = m_lookat - sbtRecord.data.position; // Do not normalize W -- it implies focal length
        //W = make_float3(1.0f, 0.0f, 0.0f);
        float wlen = length(W);
        U = normalize(cross(W, m_up));
        V = normalize(cross(U, W));

        float m_fovY = 120.0f;
        float m_aspectRatio = 1.0f;

        float vlen = wlen * tanf(0.5f * m_fovY * M_PIf / 180.0f);
        V *= vlen;
        float ulen = vlen * m_aspectRatio;
        U *= ulen;
    }

    void packAndCopyRecordIfChanged(OptixProgramGroup& programGroup)
    {
      if(previous_orientation != orientation)
      {
        // If the orientation has changed, then update the sbtRecord's localSpace to reflect that
        //sbtRecord.localSpace = 
      }
      // Only copy the data across if it's changed
      if(previous_sbtRecordData != sbtRecord.data)
      {
        #ifdef DEBUG
        std::cout<< "ALERT: Copying device memory for camera '"<<getCameraName()<<"'."<<std::endl;
        std::cout<< "sbtSize: "<<sizeof(sbtRecord)<<std::endl;
        std::cout<< "d_record: "<<d_record<<std::endl;
        std::cout<< "proggroup: "<<programGroup<<std::endl;
        std::cout<< "Position: ("<<sbtRecord.data.position.x<<", "
                                 <<sbtRecord.data.position.y<<", "
                                 <<sbtRecord.data.position.z<<")"<<std::endl;
        std::cout<< "LocalSpace: (("<<sbtRecord.data.localSpace.xAxis.x<<", "
                                    <<sbtRecord.data.localSpace.xAxis.y<<", "
                                    <<sbtRecord.data.localSpace.xAxis.z<<")"<<std::endl << "              "
                                    <<sbtRecord.data.localSpace.yAxis.x<<", "
                                    <<sbtRecord.data.localSpace.yAxis.y<<", "
                                    <<sbtRecord.data.localSpace.yAxis.z<<")"<<std::endl << "              "
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
    }

    void forcePackAndCopyRecord(OptixProgramGroup& programGroup)
    {
      #ifdef DEBUG
      std::cout<< "ALERT: FORCE Copying device memory for camera '"<<getCameraName()<<"'."<<std::endl;
      std::cout<< "d_record: "<<d_record<<std::endl;
      std::cout<< "proggroup: "<<programGroup<<std::endl;
      std::cout<< "Position: ("<<sbtRecord.data.position.x<<", "
                               <<sbtRecord.data.position.y<<", "
                               <<sbtRecord.data.position.z<<")"<<std::endl;
      std::cout<< "LocalSpace: (("<<sbtRecord.data.localSpace.xAxis.x<<", "
                                  <<sbtRecord.data.localSpace.xAxis.y<<", "
                                  <<sbtRecord.data.localSpace.xAxis.z<<")"<<std::endl << "              "
                                  <<sbtRecord.data.localSpace.yAxis.x<<", "
                                  <<sbtRecord.data.localSpace.yAxis.y<<", "
                                  <<sbtRecord.data.localSpace.yAxis.z<<")"<<std::endl << "              "
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
    //RaygenPosedContainerRecord<T> sbtRecord;
    RaygenRecord<RaygenPosedContainer<T>> sbtRecord; // The sbtRecord associated with this camera
    sutil::Quaternion orientation;

  private:
    CUdeviceptr d_record = 0;// Stores the pointer to the SBT record 

    // Change tracking duplicates (done by keeping an old copy and comparing)
    RaygenPosedContainer<T> previous_sbtRecordData;
    sutil::Quaternion previous_orientation;

    void allocateRecord()
    {
      #ifdef DEBUG
      std::cout << "Allocating camera SBT record on device. (size: "<< sizeof(sbtRecord) << ")" << std::endl;
      #endif
      if(d_record != 0)
      {
        #ifdef DEBUG
        std::cout << "WARN: Attempt to allocate camera SBT record was made when one is already allocated." << std::endl;
        #endif
        return;
      }

      CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_record ), sizeof(sbtRecord)) );
    }
    void freeRecord()
    {
      #ifdef DEBUG
      std::cout << "Freeing camera SBT record..." << std::endl;
      #endif
      CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_record)) );
    }
};
