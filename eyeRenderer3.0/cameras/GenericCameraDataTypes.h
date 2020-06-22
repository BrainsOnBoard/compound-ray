// The root file for the Record Data Types for cameras heirarchy

#pragma once

// Define template Record type for SBT records:
template <typename T>
struct RaygenRecord
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
  float3 position;
};

// Define a typedef for a generic Empty Record
//struct EmptyData {};
//typedef RaygenRecord<EmptyData> EmptyRecord;
