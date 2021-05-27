#ifndef SINES_GENERATOR_H
#define SINES_GENERATOR_H

#include "EyeGenerator.h"

#include <cmath>
#include <optixu/optixu_vector_types.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include <chrono>
#include <thread>

using namespace optix;
using namespace std;

class SinesGenerator : public EyeGenerator {
  public:
    static void sinesUdulator(SinesGenerator* sg);

    // Constructor and destructor
    SinesGenerator(int gridSize);
    virtual ~SinesGenerator();

    // EyeGenerator function overriders
    StaticCoordinate getCoordinateInfo(int i);
    void stop();
    bool hasNewDataReady();

    // Specialised Configuration Variables
    float separation;

  private:
    StaticCoordinate* coordinates;
    int size;
    int coordinateCount;
    bool stopFlag;
    bool newDataReadyFlag;
};

#endif
