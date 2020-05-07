// A header file to contain the SBT data definitions for the test object
#include "SbtRecord.h"

struct TestObjectData
{
  float r, g, b;
};

typedef SbtRecord<TestObjectData> TestObjectSbtRecord;
