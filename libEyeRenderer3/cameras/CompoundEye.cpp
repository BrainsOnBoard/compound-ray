#include "CompoundEye.h"
#include "curand_kernel.h"

CompoundEyePosedDataRecord CompoundEye::s_compoundSbtRecord;
CUdeviceptr CompoundEye::s_d_compoundRecord = 0;
OptixShaderBindingTable* CompoundEye::s_compoundSBTptr = nullptr;
OptixProgramGroup* CompoundEye::s_compoundProgramGroupPtr = nullptr;

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

void CompoundEye::InitiateCompoundRecord(OptixShaderBindingTable* sbtPtr, OptixProgramGroup* compoundProgramGroupPtr)
{
  // Store link to the shader binding table
  s_compoundSBTptr = sbtPtr;

  // Store link to the shader compound group
  s_compoundProgramGroupPtr = compoundProgramGroupPtr;

  // Initiate compound record on device VRAM
  #ifdef DEBUG
  std::cout << "Allocating compound SBT record on device (size: " << sizeof(s_compoundSbtRecord) << ")..." << std::endl;
  #endif
  if(s_d_compoundRecord != 0)
  {
    #ifdef DEBUG
    std::cout << "\tWARN: Attempt to allocate compound SBT record was made when one is already allocated." << std::endl;
    #endif
    return;
  }
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&s_d_compoundRecord), sizeof(s_compoundSbtRecord)) );
  #ifdef DEBUG
  printf("\t...allocated at %p\n", s_d_compoundRecord);
  #endif
}
void CompoundEye::FreeCompoundRecord()
{
  #ifdef DEBUG
  std::cout << "Freeing compound SBT record..." << std::endl;
  #endif
  if(s_d_compoundRecord != 0)
  {
    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(s_d_compoundRecord)) );
  }
}

void CompoundEye::forcePackAndCopyRecord(OptixProgramGroup& programGroup)
{
  // Perform the original record placement on the rendering pipeline
  DataRecordCamera<CompoundEyeData>::forcePackAndCopyRecord(programGroup);

  // Copy the contents of the current sbt Record
  s_compoundSbtRecord = sbtRecord;

  // Now perfom an injected record placement into the compound pipeline
  #ifdef DEBUG
  std::cout << "\tPerforming injected compound eye record updating..." << std::endl;
  printf("\t\td_record: %p\n", s_d_compoundRecord);
  std::cout << "\t\tproggroup: "<< *s_compoundProgramGroupPtr << std::endl;
  std::cout << "\t\tPosition: ("<<s_compoundSbtRecord.data.position.x<<", "
                               <<s_compoundSbtRecord.data.position.y<<", "
                               <<s_compoundSbtRecord.data.position.z<<")"<<std::endl;
  std::cout << "\t\tLocalSpace: (("<<s_compoundSbtRecord.data.localSpace.xAxis.x<<", "
                                  <<s_compoundSbtRecord.data.localSpace.xAxis.y<<", "
                                  <<s_compoundSbtRecord.data.localSpace.xAxis.z<<")"<<std::endl << "\t\t              "
                                  <<s_compoundSbtRecord.data.localSpace.yAxis.x<<", "
                                  <<s_compoundSbtRecord.data.localSpace.yAxis.y<<", "
                                  <<s_compoundSbtRecord.data.localSpace.yAxis.z<<")"<<std::endl << "\t\t              "
                                  <<s_compoundSbtRecord.data.localSpace.zAxis.x<<", "
                                  <<s_compoundSbtRecord.data.localSpace.zAxis.y<<", "
                                  <<s_compoundSbtRecord.data.localSpace.zAxis.z<<")"<<std::endl;
  std::cout << "\t\tOmmatidial count: " << s_compoundSbtRecord.data.specializedData.ommatidialCount << std::endl;
  std::cout << "\t\tEye Index: " << s_compoundSbtRecord.data.specializedData.eyeIndex << std::endl;
  std::cout << "\t\tSamples per Ommatidium: " << s_compoundSbtRecord.data.specializedData.samplesPerOmmatidium << std::endl;
  #endif
  // Actually pack and copy in the compound sbt record
  OPTIX_CHECK( optixSbtRecordPackHeader(*s_compoundProgramGroupPtr, &s_compoundSbtRecord) );
  CUDA_CHECK( cudaMemcpy(
              reinterpret_cast<void*>( s_d_compoundRecord ),
              &s_compoundSbtRecord,
              sizeof(s_compoundSbtRecord),
              //sizeof(CompoundEyePosedDataRecord),
              cudaMemcpyHostToDevice
              ) );
  // TODO: Try to do the same thing as the m_sbt.raygenRecord = s_d_compoundRecord bit here
  s_compoundSBTptr->raygenRecord = s_d_compoundRecord;
}
