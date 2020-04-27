#include "NonPlanarCoordinate.h"

NonPlanarCoordinate::NonPlanarCoordinate(void)
{
  //cout << "NonPlanarCoordinate made" << endl;
}
NonPlanarCoordinate::~NonPlanarCoordinate(void)
{
  //cout << "NonPlanarCoordinate destroyed" << endl;
}

// Generic helper functions and statics
float NonPlanarCoordinate::randRange(float min, float max) // TODO: Could this be inline'd?
{
  return static_cast<float>(rand()/static_cast<float>(RAND_MAX)) * (max-min) + min;
}

const float3 NonPlanarCoordinate::VERTICAL = make_float3(0.0f, 0.0f, 1.0f);
const float3 NonPlanarCoordinate::TRUE_VERTICAL = make_float3(0.0f, 1.0f, 0.0f);

// A generic distanceenergy calculator using Reiz S-energy relying on per-implementation distance calculation
float NonPlanarCoordinate::getEnergy(NonPlanarCoordinate* others[], int count, int proximity)
{
  int i,o;
  float temp;

  float nClosest[proximity];
  for(i = 0; i<proximity; i++)
    nClosest[i] = -1.0f;

  for(i = 0; i<count; i++)
  {
    NonPlanarCoordinate* sc = others[i];
    if(sc == this)
      continue; // Skip comparing with itself

    // Claculate the energyDistance to each of them
    float energyDistance = 1.0f/this->getFastDistanceTo(sc);// The denominator could be raised to a power (but is not here)

    if(energyDistance > nClosest[proximity-1])// If it's got a higher energy than the lowest-energy closest coordinate
    {
      nClosest[proximity-1] = energyDistance;// Remove the previous lowest-energy closest
      // Then iterate down the array, shuffling this energyDistance into position using an in-line sort.
      for(o = proximity-1; o>0; o--)
      {
        if(nClosest[o-1] < energyDistance)
        {
          temp = nClosest[o-1];
          nClosest[o-1] = nClosest[o];
          nClosest[o] = temp;
        }
      }
    }
  }

  float totalEnergy = nClosest[0];
  for(i = 1; i<proximity; i++)
    totalEnergy += nClosest[i];
    
  return(totalEnergy);
}
