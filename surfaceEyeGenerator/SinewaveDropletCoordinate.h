#ifndef SINEWAVE_DROPLET_COORDINATE_H
#define SINEWAVE_DROPLET_COORDINATE_H

#include "NonPlanarCoordinate.h"
#include <stdlib.h>
#include <sutil/vec_math.h>

class SinewaveDropletCoordinate : public NonPlanarCoordinate {
  public:
    static float scale;
    static float time; // in seconds, since the start.

    // Constructor/destructor
    SinewaveDropletCoordinate();
    ~SinewaveDropletCoordinate();

    // Virtual Overriders
    void randomMove(float scale);
    //float getEnergy(NonPlanarCoordinate* others[], int count, int proximity);
    void backtrack();
    StaticCoordinate getStaticCoord();
    float getFastDistanceTo(NonPlanarCoordinate* other);
    float getDistanceTo(NonPlanarCoordinate* other);

    // Unique members
    // the wave details go here.
  private:
    float2 position, oldPosition; // on a -0.5 to 0.5 grid
    float wavelength, amplitude;
    static float2 BOUNDS; // A basic bounds vector that defines the maximum distances (is all 0.5f's)
    static float2 ORIGIN; // An origin vector.
};
#endif
