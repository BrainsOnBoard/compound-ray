#ifndef NONPLANARCOORDINATE_H
#define NONPLANARCOORDINATE_H

#include <iostream>
#include <cuda_runtime.h>

struct StaticCoordinate // This should probably include a bit for the spread function control.
{
//#if defined(__cplusplus)
//  typedef optix::float3 float3;
//#endif
  float3 position;
  float3 direction;
  int padding[2];// padding so this structure is 32 bytes.
};

class NonPlanarCoordinate {
  public:

    // Constructor and destructor
    NonPlanarCoordinate();
    virtual ~NonPlanarCoordinate();
    // Takes a step in a random direction and distance (Scaled by `scale`)
    virtual void randomMove(float scale) = 0;
    // Calculates the energy (a function of how close other coordinates are to this coordinate)
    virtual float getEnergy(NonPlanarCoordinate* others[], int count, int proximity);
    // Returns the (fast) distance to the other coordinate from this one (usually other is of the same type)
    virtual float getFastDistanceTo(NonPlanarCoordinate* other) = 0;
    // Returns the true distance to the other coordinate (may incur additional processing steps)
    virtual float getDistanceTo(NonPlanarCoordinate* other) = 0;
    virtual void backtrack() = 0; // Backtracks the last step
    virtual StaticCoordinate getStaticCoord() = 0; // Returns this coordinate as a static coorinate.
  protected:
    // Static functions
    static float randRange(float min, float max);
    static const float3 VERTICAL;// = make_float(0.0f,0.0f,1.0f);
    static const float3 TRUE_VERTICAL;
    //  Some internal state that tracks position
};

#endif
