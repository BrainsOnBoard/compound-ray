#include "GenericCamera.h"


#include "ThreeSixtyCameraDataTypes.h"

class ThreeSixtyCamera : public GenericCamera {
  public:
    static const int PROGRAM_GROUP_ID = 2;// Still a horrible hacky work around

    ThreeSixtyCamera();
    ~ThreeSixtyCamera();

    void allocateRecord();
    void packAndCopyRecord(OptixProgramGroup& programGroup);

  private:
    ThreeSixtyCameraData sbtRecord;
    const size_t recordSize = sizeof(PerspectiveCameraRecord);
};
