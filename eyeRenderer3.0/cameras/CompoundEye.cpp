#include "CompoundEye.h"

CompoundEye::CompoundEye(const std::string name, size_t ommatidialCount) : DataRecordCamera<CompoundEyeData>(name)
{
  // Assign VRAM for the compound eye
  specializedData.ommatidialCount = ommatidialCount;
  allocateOmmatidialMemory();
}
CompoundEye::~CompoundEye()
{
  // Free VRAM of compound eye structure information
}

const CompactCompoundEyeData CompoundEye::getCompactData() const
{
  CompactCompoundEyeData out;
  out.compoundEyeData = specializedData;
  out.position = sbtRecord.data.position;
  out.localSpace = ls;
  return out;
}
void CompoundEye::assignOmmatidia(Ommatidium* ommatidia)
{
  CUDA_CHECK( cudaMemcpy(
              reinterpret_cast<void*>(specializedData.d_ommatidialArray),
              ommatidia,
              sizeof(Ommatidium)*specializedData.ommatidialCount,
              cudaMemcpyHostToDevice
              )
            );
}

void CompoundEye::allocateOmmatidialMemory()
{
  size_t memSize = sizeof(Ommatidium)*specializedData.ommatidialCount;
  #ifdef DEBUG
  std::cout << "Allocating ommatidial data on device. (size: "<<memSize<<")"<<std::endl;
  #endif
  if(specializedData.d_ommatidialArray != 0)
  {
    #ifdef DEBUG
    std::cout << "  NOTE: ommatidial data already allocated on device. Deallocating original data, reallocating with new." << std::endl;
    #endif
    freeOmmatidialMemory();
  }
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &(specializedData.d_ommatidialArray) ), memSize) );
}
void CompoundEye::freeOmmatidialMemory()
{
  #ifdef DEBUG
  std::cout << "Freeing ommatidial memory..." << std::endl;
  #endif
  CUDA_CHECK( cudaFree(reinterpret_cast<void*>(specializedData.d_ommatidialArray)) );
}
