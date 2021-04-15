#include "CompoundEye.h"
#include "curand_kernel.h"

CompoundEye::CompoundEye(const std::string name, const std::string shaderName, size_t ommatidialCount) : DataRecordCamera<CompoundEyeData>(name), shaderName(NAME_PREFIX + shaderName)
{
  // Assign VRAM for the compound eye
  specializedData.ommatidialCount = ommatidialCount;
  allocateOmmatidialMemory();
  // Assign VRAM for the random states
  allocateOmmatidialRandomStates();
}
CompoundEye::~CompoundEye()
{
  // Free VRAM of compound eye structure information
  freeOmmatidialMemory();
  // Free VRAM of the compound eye random states
  freeOmmatidialRandomStates();
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
    std::cout << "  NOTE: Ommatidial data already allocated on device. Deallocating original data, reallocating with new." << std::endl;
    #endif
    freeOmmatidialMemory();
  }
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &(specializedData.d_ommatidialArray) ), memSize) );
  #ifdef DEBUG
  printf("  ...allocated at %p\n", specializedData.d_ommatidialArray);
  #endif
}
void CompoundEye::freeOmmatidialMemory()
{
  #ifdef DEBUG
  std::cout << "Freeing ommatidial memory...";
  #endif
  CUDA_CHECK( cudaFree(reinterpret_cast<void*>(specializedData.d_ommatidialArray)) );
  #ifdef DEBUG
  std::cout << "  freed!" << std::endl;
  #endif
}

void CompoundEye::allocateOmmatidialRandomStates()
{
  size_t memSize = sizeof(curandState)*specializedData.ommatidialCount;
  #ifdef DEBUG
  std::cout << "Allocating per-ommatidium random states on device. (size: "<<memSize<<")"<<std::endl;
  #endif
  if(specializedData.d_randomStates != 0)
  {
    #ifdef DEBUG
    std::cout << "  NOTE: Ommatidial random states already allocated on device. Deallocating original data, reallocating with new." << std::endl;
    #endif
    freeOmmatidialRandomStates();
  }
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &(specializedData.d_randomStates) ), memSize) );
  #ifdef DEBUG
  printf("  ...allocated at %p\n", specializedData.d_randomStates);
  #endif

  // TODO: The randomStateBuffer is currently unitialized. For now we'll be initializing it with if statements in the ommatidial shader, but in the future a CUDA function could be called here to initialize it.
}
void CompoundEye::freeOmmatidialRandomStates()
{
  #ifdef DEBUG
  std::cout << "Freeing ommatidial random states...";
  #endif
  CUDA_CHECK( cudaFree(reinterpret_cast<void*>(specializedData.d_randomStates)) );
  #ifdef DEBUG
  std::cout << "  freed!" << std::endl;
  #endif
}

void CompoundEye::setSamplesPerOmmatidium(int32_t s)
{
  specializedData.samplesPerOmmatidium = max(static_cast<int32_t>(1),s);
  #ifdef DEBUG
  std::cout << "Set samples per ommatidium to " << specializedData.samplesPerOmmatidium << std::endl;
  #endif
}
void CompoundEye::changeSamplesPerOmmatidiumBy(int32_t d)
{
  setSamplesPerOmmatidium(specializedData.samplesPerOmmatidium + d);
}
