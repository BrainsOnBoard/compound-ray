#pragma once

#include <optix_stubs.h>

#include <sutil/sutilapi.h>
#include <sutil/vec_math.h>
#include "DataRecordCamera.h"
#include <sutil/Exception.h>

#include "CompoundEyeDataTypes.h"

class CompoundEye : public DataRecordCamera<CompoundEyeData> {
  public:
    CompoundEye(const std::string name, const std::string shaderName, size_t ommatidialCount);
    ~CompoundEye();

    const char* getEntryFunctionName() const { return shaderName.c_str(); }

    void copyOmmatidia(Ommatidium* ommatidia);
    const size_t getOmmatidialCount() const { return specializedData.ommatidialCount; }
    void setCompoundIndex(uint32_t index) { specializedData.eyeIndex = index; }

  private:
    static constexpr char* NAME_PREFIX = "__raygen__compound_projection_";
    const std::string shaderName;
    void allocateOmmatidialMemory();
    void freeOmmatidialMemory();
};
