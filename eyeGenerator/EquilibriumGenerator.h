#ifndef EQUILIBRIUM_GENERATOR_H
#define EQUILIBRIUM_GENERATOR_H
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <stdio.h>
#include <time.h>
#include <chrono>

#include <algorithm>

#include "EyeGenerator.h"
#include "SphericalCoordinate.h"
#include "SinewaveDropletCoordinate.h"


class EquilibriumGenerator : public EyeGenerator {
  public:
    static void basicIterator(EquilibriumGenerator* eg);
    static void rieszSEnergyIterator(EquilibriumGenerator* eg);

    
    // Constructor and Destructor
    EquilibriumGenerator(int coordinateCount);
    ~EquilibriumGenerator();

    // EyeGenerator function overriders
    StaticCoordinate getCoordinateInfo(int i);
    void stop();
    bool hasNewDataReady();

    // Specialised Sub-Generators and subclass-related functions
    void generateSphericalCoordinates();
    void generateSinewaveDropletCoordinates();

    // Specialised configuration variables
    float stepSize;
    float coordinateProximityCount;
    float varianceCap = 0.00001f;

  private:
    NonPlanarCoordinate** coordinates;
    int coordinateCount;
    bool stopFlag;
    bool newDataReadyFlag;
    
};

#endif
