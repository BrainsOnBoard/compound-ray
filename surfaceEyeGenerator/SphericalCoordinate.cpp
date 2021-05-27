#include "SphericalCoordinate.h"

using namespace std;

SphericalCoordinate::SphericalCoordinate(int idin)//void)
{
  state = normalize(make_float3(randRange(-1.0f,1.0f),randRange(-1.0f,1.0f),randRange(-1.0f,1.0f)));
  //state = normalize(make_float3(randRange(-1.5f,0.5f), 1.0f, randRange(-1.5f,0.5f)));
  oldState.x = state.x;
  oldState.y = state.y;
  oldState.z = state.z;
  id = idin + 1;
  //std::cout << "Spherical Coordinate object created" << std::endl;
}
SphericalCoordinate::~SphericalCoordinate(void)
{
  //std::cout << "Spherical Coordinate object destroyed" << std::endl;
}

inline float toDegrees(float rads)
{
  return((180*rads)/M_PI);
}

void SphericalCoordinate::randomMove(float scale)
{
  oldState = make_float3(state.x, state.y, state.z);
  /// Take a step at `stepAngle` degrees to north with `scale` distance.
  float stepAngle = randRange(0,2*M_PI);
  float stepDistance = randRange(0.0f,scale);
  /// Formulate the vector offset from the Vertical
  float3 offset = normalize(make_float3(cos(stepAngle)*stepDistance, sin(stepAngle)*stepDistance, 1.0f));// TODO: Note that this is a planar projection onto a sphere.
  //float3 offset = make_float3(0.0f,0.0f,1.0f);// offset points directly up (like vertical).

  // TODO: the offset vector should probably be formed by rotating the up unit vector forward by scale, then around z by stepAngle.
  // TODO: This vector can then be subject to rotation toward the ommatidial position.
  /// First generate the position of the current ommatidium coordinate

  float verticalAngle = -acos(dot(state, VERTICAL));// The vertical angle of the current state (multiplied by -1 because the upcoming rotation is ccw)

  //cout << endl << "> state vertical angle : " << toDegrees(verticalAngle) << endl;
  //cout << "> offset vertical angle: " << toDegrees(acos(dot(offset,VERTICAL))) << endl;
  //cout << "> rotating offset..."<<endl;

  float horizontalAngle;
  if (state.x == 0.0f && state.y == 0.0f)
    horizontalAngle = 0.0f;
  else
    horizontalAngle = atan2(state.y, state.x);

  // TODO: The below rotation needs to be converted into a single matrix operation.
  float3 oldOffset = make_float3(offset.x, offset.y, offset.z);
  /// Then rotate the offset vector to the position of the current ommatidium coordinate
  // Rotate about the y-axis
  offset.x = cos(verticalAngle) * oldOffset.x - sin(verticalAngle) * oldOffset.z;
  offset.z = sin(verticalAngle) * oldOffset.x + cos(verticalAngle) * oldOffset.z;
  
  // Update oldOffset for the next rotation.
  oldOffset.x = offset.x;
  oldOffset.y = offset.y;
  oldOffset.z = offset.z;

  // Rotate about the z-axis
  offset.x = cos(horizontalAngle) * oldOffset.x - sin(horizontalAngle) * oldOffset.y;
  offset.y = sin(horizontalAngle) * oldOffset.x + cos(horizontalAngle) * oldOffset.y;


  // Update the state to reflect the new offset state.
  state.x = offset.x;
  state.y = offset.y;
  state.z = offset.z;
}


float SphericalCoordinate::getEnergy(NonPlanarCoordinate* others[], int count, int proximity)
{
  int i,o;
  float temp;

  float nClosest[proximity];
  for(i = 0; i<proximity; i++)
    nClosest[i] = -1.0f;

  for(i = 0; i<count; i++)
  {
    SphericalCoordinate* sc = (SphericalCoordinate*)others[i];
    if(sc == this)
      continue; // Skip comparing with itself

    // Claculate the energyDistance to each of them
    float energyDistance = 1.0f/this->getFastDistanceTo(sc);// The denominator could be raised to a power (but is not here)
    
    // Generate a nonuniform distribution (Ha! The irony!)
    //float energyOffset = abs(dot(normalize(make_float3(sc->state.x, 0.0f, sc->state.z)), sc->state));
    //energyDistance /= 10.0f*energyOffset*energyOffset*energyOffset;

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

void SphericalCoordinate::backtrack()
{
  //std::cout << "Backtracking..." << std::endl;
  state.x = oldState.x;
  state.y = oldState.y;
  state.z = oldState.z;
}

int SphericalCoordinate::getId()
{
  return id;
}

// Returns the actual distance
float SphericalCoordinate::getDistanceTo(NonPlanarCoordinate* other)
{
  return getFastDistanceTo(other) * SphericalCoordinate::radius;
}
// Returns the angular distance, but not the actual distance around the sphere (because it's faster and doesn't vary between coordinates)
float SphericalCoordinate::getFastDistanceTo(NonPlanarCoordinate* other)
{
  float angularDistance = acos(dot(this->state, ((SphericalCoordinate*)other)->state));
  if(isnan(angularDistance))
    return (0.0f);// Return zero as default behaviour
  return(angularDistance);
}

void SphericalCoordinate::cloneTo(SphericalCoordinate* clone)
{
  clone->state.x = state.x;
  clone->state.y = state.y;
  clone->state.z = state.z;
}

StaticCoordinate SphericalCoordinate::getStaticCoord()
{
  StaticCoordinate sc;
  sc.direction = state;
  sc.position = state * radius;
  return sc;
}

float SphericalCoordinate::radius = 5;

/*inline float SphericalCoordinate::getClosestDistance(NonPlanarCoordinate* others[], int count)
{
  return (radius * getCloasestDistanceFast(others, count));
}
// Returns distance as an angle as it's faster, the slow version will just scale it properly.
float SphericalCoordinate::getClosestDistanceFast(NonPlanarCoordinate* others[], int count)
{
  int i,o;
  float temp;
  //cout << "Vector " << this->getId() << ": (" <<  this->state.x << ", " << this->state.y << ", " << this->state.z << ")" << endl;

  const int CLOSEST_COUNT = 3;
  float nClosest[CLOSEST_COUNT];
  for(i = 0; i<CLOSEST_COUNT; i++)
    nClosest[i] = std::numeric_limits<float>::max();

  for(i = 0; i<count; i++)
  {
    SphericalCoordinate* sc = (SphericalCoordinate*)others[i];
    if(sc == this)
      continue; // Skip comparing with itself

    // Claculate the distance to each of them
    float distance = this->getFastDistanceTo(sc);

    if(distance <= nClosest[CLOSEST_COUNT-1])// If it's closer than the furthest closest
    {
      nClosest[CLOSEST_COUNT-1] = distance;// Remove the previous furthest closest
      // Then iterate down the array, shuffling this distance into position using an in-line sort.
      for(o = CLOSEST_COUNT-1; o>0; o--)
      {
        if(nClosest[o-1] > distance)
        {
          temp = nClosest[o-1];
          nClosest[o-1] = nClosest[o];
          nClosest[o] = temp;
        }
      }
    }
  }

  float sumClosestDistance = nClosest[0];
  for(i = 1; i<CLOSEST_COUNT; i++)
    sumClosestDistance *= nClosest[i];
    
  return(sumClosestDistance);
}
*/
