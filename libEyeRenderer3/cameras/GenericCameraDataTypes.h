// The root file for the Record Data Types for cameras heirarchy

#pragma once

// Define template Record type for SBT records:
template <typename T>
struct RaygenRecord
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

inline bool operator==(const float3 l, const float3 r)
{ return (l.x == r.x && l.y == r.y && l.z == r.z); }
inline bool operator==(const float2 l, const float2 r)
{ return (l.x == r.x && l.y == r.y); }
struct LocalSpace
{
  float3 xAxis = {1.0f, 0.0f, 0.0f};
  float3 yAxis = {0.0f, 1.0f, 0.0f};
  float3 zAxis = {0.0f, 0.0f, 1.0f};
  inline bool operator==(const LocalSpace& r)
  { return (this->xAxis == r.xAxis && this->yAxis == r.yAxis && this->zAxis == r.zAxis); }
  inline bool operator!=(const LocalSpace& r)
  { return !(*this==r); }
  //inline float3 transform(const float3 v)
  //{
  //  return(v.x*xAxis + v.y*yAxis + v.z*zAxis);
  //}
};
template <typename T>
struct RaygenPosedContainer
{
  T specializedData;
  float3 position = {0.0f, 0.0f, 0.0f};
  LocalSpace localSpace;

  inline bool operator==(const RaygenPosedContainer<T>& r)
  { return (this->position == r.position && this->localSpace == r.localSpace && this->specializedData == r.specializedData); }
  inline bool operator!=(const RaygenPosedContainer<T>& r)
  { return !(*this==r); }
};

//A generic record type that stores generic specialised user data alongside pose data
template <typename T>
using RaygenPosedContainerRecord = RaygenRecord<RaygenPosedContainer<T>>;
