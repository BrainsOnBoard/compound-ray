#include "GenericCamera.h"

#include <optix_stubs.h>// Needed for optixSbtRecordPackHeader


#include "ThreeSixtyCameraDataTypes.h"

class ThreeSixtyCamera : public GenericCamera {
  public:
    static const int PROGRAM_GROUP_ID = 2;// Still a horrible hacky work around

    ThreeSixtyCamera();
    ~ThreeSixtyCamera();

    void allocateRecord();
    void packAndCopyRecord(OptixProgramGroup& programGroup);

  private:
    ThreeSixtyCameraRecord sbtRecord;
    const size_t recordSize = sizeof(ThreeSixtyCameraRecord);
};
