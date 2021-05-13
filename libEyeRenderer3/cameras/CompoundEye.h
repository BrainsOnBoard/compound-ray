#pragma once

#include <optix_stubs.h>

#include <sutil/sutilapi.h>
#include <sutil/vec_math.h>
#include "DataRecordCamera.h"
#include <sutil/Exception.h>

#include "CompoundEyeDataTypes.h"

class CompoundEye : public DataRecordCamera<CompoundEyeData> {
  public:
    static void InitiateCompoundRecord(OptixShaderBindingTable& compoundSbt, OptixProgramGroup& compoundProgramGroup, const CUdeviceptr& targetRecord);
    static void FreeCompoundRecord();
    static void RedirectCompoundDataPointer(OptixProgramGroup& programGroup, const CUdeviceptr& targetRecord);

    CompoundEye(const std::string name, const std::string shaderName, size_t ommatidialCount);
    ~CompoundEye();

    const char* getEntryFunctionName() const { return shaderName.c_str(); }

    void copyOmmatidia(Ommatidium* ommatidia);
    const size_t getOmmatidialCount() const { return specializedData.ommatidialCount; }

    const uint32_t getSamplesPerOmmatidium() const { return specializedData.samplesPerOmmatidium; }
    void setSamplesPerOmmatidium(int32_t s);
    void changeSamplesPerOmmatidiumBy(int32_t d);

  private:
    // Static consts for configuration
    static constexpr const char* NAME_PREFIX = "__raygen__compound_projection_";

    // Static variables for management of the compound pipeline's single redirecting record
    static RecordPointerRecord s_compoundRecordPtrRecord;
    static CUdeviceptr s_d_compoundRecordPtrRecord;


    const std::string shaderName;
    void allocateOmmatidialMemory();
    void allocateOmmatidialRandomStates();
    void allocateCompoundRenderingBuffer();
    void freeOmmatidialMemory();
    void freeOmmatidialRandomStates();
    void freeCompoundRenderingBuffer();
};
