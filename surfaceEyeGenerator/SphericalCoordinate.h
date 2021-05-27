#ifndef SPHERICALCOORDINATE_H
#define SPHERICALCOORDINATE_H

#include "NonPlanarCoordinate.h"

#include <iostream>
#include <limits>
#include <cmath>
#include <stdlib.h>

#include <sutil/vec_math.h>

class SphericalCoordinate : public NonPlanarCoordinate {
  public:
    // static info for this setup
    static float radius;

    // Constructor/destructor
    SphericalCoordinate(int idin);
    ~SphericalCoordinate();

    // Virtual Overriders
    void randomMove(float scale);
    float getEnergy(NonPlanarCoordinate* others[], int count, int proximity);
    void backtrack();
    StaticCoordinate getStaticCoord();
    float getFastDistanceTo(NonPlanarCoordinate* other);
    float getDistanceTo(NonPlanarCoordinate* other);

    // Unique functions
    int getId();
    //float getClosestDistance(NonPlanarCoordinate* others[], int count);
    //float getClosestDistanceFast(NonPlanarCoordinate* others[], int count);

    void cloneTo(SphericalCoordinate* clone);

  private:
    int id;
    float3 state;// Stores the actual directional state of this object
    float3 oldState;// Stores the last directional state for backtracking

    void setLatLong(float lat, float lon);
};

#endif
