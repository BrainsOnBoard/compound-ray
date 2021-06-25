#pragma once

#include <optix_stubs.h>
#include <string>

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

    CompoundEye(const std::string name, const std::string shaderName, size_t ommatidialCount, const std::string& eyeDataPath);
    ~CompoundEye();

    const char* getEntryFunctionName() const { return shaderName.c_str(); }

    void setOmmatidia(Ommatidium* ommatidia, size_t count); // Copies in the ommatidial list, resetting and reallocating all affected memory if count differs from the current ommatidial count
    void copyOmmatidia(Ommatidium* ommatidia); // Copies in the ommatidial list given, to the length of the current number of ommatidia in the eye
    const size_t getOmmatidialCount() const { return specializedData.ommatidialCount; }

    const uint32_t getSamplesPerOmmatidium() const { return specializedData.samplesPerOmmatidium; }
    void setSamplesPerOmmatidium(int32_t s);
    void changeSamplesPerOmmatidiumBy(int32_t d);
    void setShaderName(const std::string shaderName);

    // Sets this eye's randomsConfigured to true. TODO(RANDOMS): Will not be required when randoms are ensured configured on creation. Literally only used in libEyeRenderer's renderFrame function:
    void setRandomsAsConfigured() { specializedData.randomsConfigured = true; } 

    std::string eyeDataPath; // A string containing the path to the eye data (note: easily mutable)

  private:
    // Static consts for configuration
    static constexpr const char* NAME_PREFIX = "__raygen__compound_projection_";

    // Static variables for management of the compound pipeline's single redirecting record
    static RecordPointerRecord s_compoundRecordPtrRecord;
    static CUdeviceptr s_d_compoundRecordPtrRecord;

    // Changes the ommatidial count, resetting ommatidial, random
    // and compound rendering buffers if the count has changed
    void reconfigureOmmatidialCount(size_t count);

    std::string shaderName;
    void allocateOmmatidialMemory();
    void allocateOmmatidialRandomStates();
    void allocateCompoundRenderingBuffer();
    void freeOmmatidialMemory();
    void freeOmmatidialRandomStates();
    void freeCompoundRenderingBuffer();
};
