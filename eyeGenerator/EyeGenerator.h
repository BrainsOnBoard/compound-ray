#ifndef EYE_GENERATOR_H
#define EYE_GENERATOR_H

#include "NonPlanarCoordinate.h"// Include it for the StaticCoordinate type too.

class EyeGenerator {
  public:
    // Constructor and destructor
    EyeGenerator();
    virtual ~EyeGenerator();
    // Allows access to individual coordinate info
    virtual StaticCoordinate getCoordinateInfo(int i) = 0;
    // Stops all threaded action of this object
    virtual void stop() = 0;
    // Allows the generator to be polled to see if it's got any new data
    virtual bool hasNewDataReady() = 0;
};

#endif
