#include "CompoundEye.h"
#include "curand_kernel.h"

RecordPointerRecord CompoundEye::s_compoundRecordPtrRecord;
CUdeviceptr CompoundEye::s_d_compoundRecordPtrRecord = 0;

CompoundEye::CompoundEye(const std::string name, const std::string shaderName, size_t ommatidialCount) : DataRecordCamera<CompoundEyeData>(name), shaderName(NAME_PREFIX + shaderName)
{
  // Assign VRAM for the compound eye
  specializedData.ommatidialCount = ommatidialCount;
  specializedData.samplesPerOmmatidium = 30;
  specializedData.ommatidialCount = 42;
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
  std::cout << "Freeing ommatidial memory..."<<std::endl;
  #endif
  CUDA_CHECK( cudaFree(reinterpret_cast<void*>(specializedData.d_ommatidialArray)) );
  specializedData.d_ommatidialArray = 0;
  #ifdef DEBUG
  std::cout << "Ommatidial memory freed!" << std::endl;
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
}
void CompoundEye::freeOmmatidialRandomStates()
{
  #ifdef DEBUG
  std::cout << "Freeing ommatidial random states..." << std::endl;
  #endif
  CUDA_CHECK( cudaFree(reinterpret_cast<void*>(specializedData.d_randomStates)) );
  specializedData.d_randomStates = 0;
  #ifdef DEBUG
  std::cout << "Ommatidial random states freed!" << std::endl;
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


// ----------------------------------------------------------------
//    Compound record handling
// ----------------------------------------------------------------

void CompoundEye::InitiateCompoundRecord(OptixShaderBindingTable& compoundSbt, OptixProgramGroup& compoundProgramGroup, const CUdeviceptr& targetRecord)
{
  // Allocate compound record (pointer to a camera) on device VRAM
  #ifdef DEBUG
  std::cout << "Allocating compound SBT pointer record on device (size: " << sizeof(s_compoundRecordPtrRecord) << ")..." << std::endl;
  #endif
  if(s_d_compoundRecordPtrRecord != 0)
  {
    #ifdef DEBUG
    std::cout << "\tWARN: Attempt to allocate compound SBT pointer record was made when one is already allocated." << std::endl;
    #endif
    return;
  }
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&s_d_compoundRecordPtrRecord), sizeof(s_compoundRecordPtrRecord)) );
  #ifdef DEBUG
  printf("\t...allocated at %p\n", s_d_compoundRecordPtrRecord);
  #endif

  // Actually point the record to the target record
  // and update the VRAM to reflect this change
  // TODO: Replace the pointer below with a reference
  RedirectCompoundDataPointer(compoundProgramGroup, targetRecord);
  
  std::cout << "Data redirected, setting record... ";
  // Bind the record to the SBT
  compoundSbt.raygenRecord = s_d_compoundRecordPtrRecord;
  std::cout  << "done!" << std::endl;

}
void CompoundEye::FreeCompoundRecord()
{
  #ifdef DEBUG
  std::cout << "Freeing compound SBT record... ";
  #endif
  if(s_d_compoundRecordPtrRecord!= 0)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(s_d_compoundRecordPtrRecord)) );
    s_d_compoundRecordPtrRecord = 0;
    #ifdef DEBUG
    std::cout << "done!" << std::endl;
    #endif
  }
  #ifdef DEBUG
  else{
    std::cout << "record already freed!" << std::endl;
  }
  #endif
}

void CompoundEye::RedirectCompoundDataPointer(OptixProgramGroup& programGroup, const CUdeviceptr& targetRecord)
{
  #ifdef DEBUG
  std::cout << "Redirecting compound record pointer..." << std::endl;
  std::cout << "\tPacking header..." << std::endl;
  #endif
  OPTIX_CHECK( optixSbtRecordPackHeader(programGroup, &s_compoundRecordPtrRecord) );
  #ifdef DEBUG
  std::cout << "\tCopying to VRAM...";
  #endif
  CUDA_CHECK( cudaMemcpy(
              reinterpret_cast<void*>(s_d_compoundRecordPtrRecord),
              &s_compoundRecordPtrRecord,
              sizeof(s_compoundRecordPtrRecord),
              cudaMemcpyHostToDevice
              ) );
  #ifdef DEBUG
  std::cout << "\t...Copy complete!" << std::endl;
  printf("\tCompound record redirected to %p\n", targetRecord);
  #endif
}
