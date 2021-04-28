#include "GenericCameraDataTypes.h"

struct CompoundEyeData
{
  CUdeviceptr d_ommatidialArray = 0;// Points to a list of Ommatidium objects in VRAM
  size_t ommatidialCount;           // The number of ommatidia in this eye
  uint32_t samplesPerOmmatidium = 1;// The number of samples taken from each ommatidium for this eye
  CUdeviceptr d_randomStates = 0; // Pointer to this compound eye's random state buffer

  inline bool operator==(const CompoundEyeData& other)
  { return (this->ommatidialCount == other.ommatidialCount && this->d_ommatidialArray == other.d_ommatidialArray &&
            this->samplesPerOmmatidium == other.samplesPerOmmatidium && this->d_randomStates == other.d_randomStates); }
};

// The ommatidium object
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
  CUdeviceptr d_currentCompoundEyeRecord = 0; // Points to the current compound eye Record
  CUdeviceptr d_compoundEyes = 0;// Points to an eyeCount-long list of CUdeviceptrs pointing at compound eye records in VRAM 
  size_t eyeCount;
};
typedef RaygenRecord<CompoundEyeCollectionData> EyeCollectionRecord;

// A simple record type that stores a pointer to another on-device record, used within the compound rendering pipeline to retrieve information from the projection pipeline
struct RecordPointer
{
  CUdeviceptr d_record = 0; // Points to another record on VRAM
};
typedef RaygenRecord<RecordPointer> RecordPointerRecord;
