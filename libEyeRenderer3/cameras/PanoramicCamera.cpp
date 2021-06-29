#include "PanoramicCamera.h"

//#define DEBUG

#ifdef DEBUG
#include <iostream>
#endif

PanoramicCamera::PanoramicCamera(const std::string name) : DataRecordCamera<PanoramicCameraData>(name)
{
  // Allocate the SBT record for the associated raygen program
  #ifdef DEBUG
  std::cout << "Creating 360 camera." << std::endl;
  #endif
  // set the start radius of the 360 camera
  setStartRadius(0.0f);
  std::cout << "My d_pointer is at: " << getRecordPtr() << std::endl;
}
PanoramicCamera::~PanoramicCamera()
{
  #ifdef DEBUG
  std::cout << "Destroying 360 camera." << std::endl;
  #endif
}

void PanoramicCamera::setStartRadius(float d)
{
  sbtRecord.data.specializedData.startRadius = d;
}
