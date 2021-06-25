#include "CompoundEye.h"
#include "curand_kernel.h"

RecordPointerRecord CompoundEye::s_compoundRecordPtrRecord;
CUdeviceptr CompoundEye::s_d_compoundRecordPtrRecord = 0;

CompoundEye::CompoundEye(const std::string name, const std::string shaderName, size_t ommatidialCount, const std::string& eyeDataPath) : DataRecordCamera<CompoundEyeData>(name), shaderName(NAME_PREFIX + shaderName)
{
  //// Assign VRAM for compound eye structure configuration
  reconfigureOmmatidialCount(ommatidialCount);

  // Set this object's eyeDataPath to a copy of the given eyeDataPath
  this->eyeDataPath = std::string(eyeDataPath);

}
CompoundEye::~CompoundEye()
{
  // Free VRAM of compound eye structure information
  freeOmmatidialMemory();
  // Free VRAM of the compound eye random states
  freeOmmatidialRandomStates();
  // Free VRAM of the compound eye's rendering buffer
}

void CompoundEye::setShaderName(const std::string shaderName)
{
  this->shaderName = NAME_PREFIX + shaderName;
}

void CompoundEye::setOmmatidia(Ommatidium* ommatidia, size_t count)
{
  reconfigureOmmatidialCount(count); // Change the count and buffers (if required)
  copyOmmatidia(ommatidia); // Actually copy the data in
}
void CompoundEye::reconfigureOmmatidialCount(size_t count)
{
  // Only do this if the count has changed
  if(count != specializedData.ommatidialCount)
  {
    // Assign VRAM for compound eye structure configuration
    specializedData.ommatidialCount = count;
    allocateOmmatidialMemory();
    // Assign VRAM for the random states
    allocateOmmatidialRandomStates();
    // Assign VRAM for the compound rendering buffer
    allocateCompoundRenderingBuffer();
  }
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
  CUDA_SYNC_CHECK();
}

void CompoundEye::allocateOmmatidialMemory()
{
  size_t memSize = sizeof(Ommatidium)*specializedData.ommatidialCount;
  #ifdef DEBUG
  std::cout << "Clearing and allocating ommatidial data on device. (size: "<<memSize<<", "<<specializedData.ommatidialCount<<" blocks)"<<std::endl;
  #endif
  freeOmmatidialMemory();
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &(specializedData.d_ommatidialArray) ), memSize) );
  #ifdef DEBUG
  printf("\t...allocated at %p\n", specializedData.d_ommatidialArray);
  #endif
  CUDA_SYNC_CHECK();
}
void CompoundEye::freeOmmatidialMemory()
{
  #ifdef DEBUG
  std::cout << "[CAMERA: " << getCameraName() << "] Freeing ommatidial memory... ";
  #endif
  if(specializedData.d_ommatidialArray != 0)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(specializedData.d_ommatidialArray)) );
    specializedData.d_ommatidialArray = 0;
    #ifdef DEBUG
    std::cout << "Ommatidial memory freed!" << std::endl;
    #endif
  }
  #ifdef DEBUG
  else{
    std::cout << "Ommatidial memory already free, skipping..." << std::endl;
  }
  #endif
  CUDA_SYNC_CHECK();
}

void CompoundEye::allocateOmmatidialRandomStates()
{
  size_t blockCount = specializedData.ommatidialCount * specializedData.samplesPerOmmatidium;// The number of cuRand states
  size_t memSize = sizeof(curandState)*blockCount;
  #ifdef DEBUG
  std::cout << "[CAMERA: " << getCameraName() << "] Clearing and allocating per-ommatidium random states on device. (size: "<<memSize<<", "<<blockCount<<" blocks)"<<std::endl;
  #endif
  freeOmmatidialRandomStates();
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &(specializedData.d_randomStates) ), memSize) );
  #ifdef DEBUG
  printf("  ...allocated at %p\n", specializedData.d_randomStates);
  #endif
  // TODO(RANDOMS): The randomStateBuffer is currently unitialized. For now we'll be initializing it with if statements in the ommatidial shader, but in the future a CUDA function could be called here to initialize it.
  // Set this camera's randomsConfigured switch to false so that the aforementioned if statement can work:
  specializedData.randomsConfigured = false;
  CUDA_SYNC_CHECK();

}
void CompoundEye::freeOmmatidialRandomStates()
{
  #ifdef DEBUG
  std::cout << "[CAMERA: " << getCameraName() << "] Freeing ommatidial random states... ";
  #endif
  if(specializedData.d_ommatidialArray != 0)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(specializedData.d_randomStates)) );
    specializedData.d_randomStates = 0;
    #ifdef DEBUG
    std::cout << "Ommatidial random states freed!" << std::endl;
    #endif
  }
  #ifdef DEBUG
  else{
    std::cout << "Ommatidial random states already free, skipping..." << std::endl;
  }
  #endif
  CUDA_SYNC_CHECK();
}
void CompoundEye::allocateCompoundRenderingBuffer()
{
  size_t blockCount = specializedData.ommatidialCount * specializedData.samplesPerOmmatidium;
  size_t memSize = sizeof(float3)*blockCount;
  #ifdef DEBUG
  std::cout << "[CAMERA: " << getCameraName() << "] Clearing and allocating compound render buffer on device. (size: "<<memSize<<", "<<blockCount<<" blocks)"<<std::endl;
  #endif
  freeCompoundRenderingBuffer();
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &(specializedData.d_compoundBuffer) ), memSize) );
  #ifdef DEBUG
  printf("  ...allocated at %p\n", specializedData.d_compoundBuffer);
  #endif
  CUDA_SYNC_CHECK();
}
void CompoundEye::freeCompoundRenderingBuffer()
{
  #ifdef DEBUG
  std::cout << "[CAMERA: " << getCameraName() << "] Freeing compound render buffer... ";
  #endif
  if(specializedData.d_compoundBuffer != 0)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(specializedData.d_compoundBuffer)) );
    specializedData.d_compoundBuffer= 0;
    #ifdef DEBUG
    std::cout << "Compound buffer freed!" << std::endl;
    #endif
  }
  #ifdef DEBUG
  else{
    std::cout << "Compound buffer already free, skipping..." << std::endl;
  }
  #endif
  CUDA_SYNC_CHECK();
}

void CompoundEye::setSamplesPerOmmatidium(int32_t s)
{
  specializedData.samplesPerOmmatidium = max(static_cast<int32_t>(1),s);
  allocateOmmatidialRandomStates();
  allocateCompoundRenderingBuffer();
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
  #endif
  s_compoundRecordPtrRecord.data.d_record = targetRecord;
  #ifdef DEBUG
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
