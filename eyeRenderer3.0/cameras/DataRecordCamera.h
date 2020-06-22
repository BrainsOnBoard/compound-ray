//#include <optix.h>
//#include <sutil/Quaternion.h>
//#include <sutil/Exception.h>
//
//#include <iostream>

#define DEBUG

#include "GenericCameraDataTypes.h"
#include "GenericCamera.h"

template<typename T>
class DataRecordCamera : public GenericCamera {
  public:
    DataRecordCamera()
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

    // TODO: Rotation and position code here.
    // They will alter RaygenRecord.position and .rotation
    // Then all of this needs to be removed from Generic Camera to here
    // Then finally we can actually get the code doing the thing.

    void allocateRecord()
    {
      #ifdef DEBUG
      std::cout << "Allocating camera SBT record on device." << std::endl;
      #endif
      if(based_record != 0)
      {
        #ifdef DEBUG
        std::cout << "WARN: Attempt to allocate camera SBT record was made when one is already allocated." << std::endl;
        #endif
        return;
      }

      CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &based_record ), sizeof(T)) );
    }
    void freeRecord()
    {
      #ifdef DEBUG
      std::cout << "Freeing camera SBT record..." << std::endl;
      #endif
      CUDA_CHECK( cudaFree(reinterpret_cast<void*>(based_record)) );
    }

  protected:
    RaygenRecord<T> baseRecord; // To rename to 'sbtRecord'.
    CUdeviceptr based_record;// Stores the SBT record required for this camera
};
