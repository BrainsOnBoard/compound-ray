// This is a general definition file to be referenced by all SBT record definitions

#ifndef SBT_RECORD_H
#define SBT_RECORD_H
#include <optix.h>

template <typename T>
struct SbtRecord
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};
#endif
