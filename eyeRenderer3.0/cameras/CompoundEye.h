#pragma once

#include <optix_stubs.h>

#include <sutil/sutilapi.h>
#include <sutil/vec_math.h>
#include "DataRecordCamera.h"
#include <sutil/Exception.h>

#include "CompoundEyeDataTypes.h"

struct CompactCompoundEyeData {
  CompoundEyeData compoundEyeData;
  float3 position;
  LocalSpace localSpace;
};

class CompoundEye : public DataRecordCamera<CompoundEyeData> {
  public:
    CompoundEye(const std::string name, size_t ommatidialCount);
    ~CompoundEye();

    const char* getEntryFunctionName() const { return ("__raygen__sphericallyProjectedCompoundEye"); }

    void assignOmmatidia(Ommatidium* ommatidia);
    const size_t getOmmatidialCount() const { return specializedData.ommatidialCount; }
    const CompactCompoundEyeData getCompactData() const;
  private:
    void allocateOmmatidialMemory();
    void freeOmmatidialMemory();
};
