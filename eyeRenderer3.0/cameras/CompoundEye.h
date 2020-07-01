#pragma once

#include <optix_stubs.h>

#include <sutil/sutilapi.h>
#include <sutil/vec_math.h>
#include "DataRecordCamera.h"
#include <sutil/Exception.h>

#include "CompoundEyeDataTypes.h"

class CompoundEye : public DataRecordCamera<CompoundEyeData> {
  public:
    CompoundEye(const std::string name, size_t ommatidialCount);
    ~CompoundEye();

    const char* getEntryFunctionName() const { return ("__raygen__single_compound_eye"); }

    void assignOmmatidia(Ommatidium* ommatidia);
  private:
    void allocateOmmatidialMemory();
    void freeOmmatidialMemory();
};
