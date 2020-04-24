#ifndef EQUILIBRIUM_GENERATOR_H
#define EQUILIBRIUM_GENERATOR_H

#include "EyeGenerator.h"

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
  
  private:
    NonPlanarCoordinate** coordinates;
    int coordinateCount;
    bool stopFlag;
    bool newDataReadyFlag;
    
};

#endif
