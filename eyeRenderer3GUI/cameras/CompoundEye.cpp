#include "CompoundEye.h"

CompoundEye::CompoundEye(const std::string name, const std::string shaderName, size_t ommatidialCount) : DataRecordCamera<CompoundEyeData>(name), shaderName(NAME_PREFIX + shaderName)
{
  // Assign VRAM for the compound eye
  specializedData.ommatidialCount = ommatidialCount;
  allocateOmmatidialMemory();
}
CompoundEye::~CompoundEye()
{
  // Free VRAM of compound eye structure information
  freeOmmatidialMemory();
}

void CompoundEye::copyOmmatidia(Ommatidium* ommatidia)
{
  std::cout << "Copying data.."<<std::endl;
  CUDA_CHECK( cudaMemcpy(
              reinterpret_cast<void*>(specializedData.d_ommatidialArray),
              ommatidia,
              sizeof(Ommatidium)*specializedData.ommatidialCount,
              cudaMemcpyHostToDevice
              )
            );
  std::cout << "  ...data copied."<<std::endl;
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
  std::cout << "  ...ommatidial data allocated on device."<<std::endl;
}
void CompoundEye::freeOmmatidialMemory()
{
  #ifdef DEBUG
  std::cout << "Freeing ommatidial memory..." << std::endl;
  #endif
  CUDA_CHECK( cudaFree(reinterpret_cast<void*>(specializedData.d_ommatidialArray)) );
}

void CompoundEye::setSamplesPerOmmatidium(int32_t s)
{
  specializedData.samplesPerOmmatidium = max(static_cast<int32_t>(1),s);
  std::cout << "Set samples per ommatidium to " << specializedData.samplesPerOmmatidium << std::endl;
}
void CompoundEye::changeSamplesPerOmmatidiumBy(int32_t d)
{
  setSamplesPerOmmatidium(specializedData.samplesPerOmmatidium + d);
}
