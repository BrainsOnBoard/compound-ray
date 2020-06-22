#include "GenericCamera.h"
#include "DataRecordCamera.h"

#include <optix_stubs.h>// Needed for optixSbtRecordPackHeader


#include "ThreeSixtyCameraDataTypes.h"

class ThreeSixtyCamera : public DataRecordCamera<ThreeSixtyCameraData> {
  public:
    ThreeSixtyCamera();
    ~ThreeSixtyCamera();

    void allocateRecord();
    void packAndCopyRecord(OptixProgramGroup& programGroup);

    const char* getEntryFunctionName() const { return "__raygen__panoramic"; }

  private:
    ThreeSixtyCameraRecord sbtRecord;
    const size_t recordSize = sizeof(ThreeSixtyCameraRecord);
};
