#include "GenericCameraDataTypes.h"

struct CompoundEyeData
{
  CUdeviceptr d_ommatidialArray = 0;// Points to a list of Ommatidium objects in VRAM
  size_t ommatidialCount; 
  CUdeviceptr d_compoundArray = 0;// Points to the pre-computed compound array TODO: I'm not sure if this is actually needed?
  uint32_t eyeIndex;// This eye's index within the pre-computed compound array
  uint32_t samplesPerOmmatidium = 1;

  inline bool operator==(const CompoundEyeData& other)
  { return (this->ommatidialCount == other.ommatidialCount && this->d_ommatidialArray == other.d_ommatidialArray &&
            this->d_compoundArray == other.d_compoundArray && this->eyeIndex == other.eyeIndex &&
            this->samplesPerOmmatidium == other.samplesPerOmmatidium); }
};

struct Ommatidium
{
  float3 relativePosition;
  float3 relativeDirection;
  float acceptanceAngleRadians;
};

typedef RaygenPosedContainer<CompoundEyeData> CompoundEyePosedData;
typedef RaygenRecord<CompoundEyePosedData> CompoundEyePosedDataRecord;

// Structure for storing references to all the compound eyes
// (in their own places on VRAM) to perform simultaneous eye renders
struct CompoundEyeCollectionData
{
  CUdeviceptr d_compoundEyes = 0;// Points to a eyeCount-long list of CUdeviceptrs pointing at compound eye records in VRAM 
  size_t eyeCount;
};
typedef RaygenRecord<CompoundEyeCollectionData> EyeCollectionRecord;
