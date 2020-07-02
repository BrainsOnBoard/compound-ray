#include "GenericCameraDataTypes.h"

struct CompoundEyeData
{
  CUdeviceptr d_ommatidialArray = 0;// Points to a list of Ommatidium objects in VRAM
  size_t ommatidialCount; 

  inline bool operator==(const CompoundEyeData& other)
  { return (this->ommatidialCount == other.ommatidialCount && this->d_ommatidialArray == other.d_ommatidialArray); }
};

struct Ommatidium
{
  float3 relativePosition;
  float3 relativeDirection;
  float acceptanceAngle;
};

typedef RaygenPosedContainer<CompoundEyeData> CompoundEyePosedData;
