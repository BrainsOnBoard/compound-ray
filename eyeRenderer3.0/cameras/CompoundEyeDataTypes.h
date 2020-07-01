#include "GenericCameraDataTypes.h"

struct CompoundEyeData
{
  CUdeviceptr d_ommatidialArray = 0;// Points to a list of Ommatidium objects in VRAM
  size_t ommatidialCount; 
};

struct Ommatidium
{
  float3 relativePosition;
  float3 relativeDirection;
  float acceptanceAngle;
};

typedef RaygenPosedContainer<CompoundEyeData> CompoundEyePosedData;
