#include "SinewaveDropletCoordinate.h"

using namespace std;

SinewaveDropletCoordinate::SinewaveDropletCoordinate()
{
  position = make_float2(randRange(-0.5f, 0.5f), randRange(-0.5f, 0.5f));
  oldPosition.x = position.x;
  oldPosition.y = position.y;
  wavelength = 1.0f;
  amplitude = 1.0f;
}
SinewaveDropletCoordinate::~SinewaveDropletCoordinate(void) { }

void SinewaveDropletCoordinate::randomMove(float scale)
{
  // Save the old state
  oldPosition.x = position.x;
  oldPosition.y = position.y;
  // Makes a random move in a random direction
  float angle = randRange(0, M_PI*2.0f);
  float distance = randRange(0, scale);
  position.x += cos(angle) * distance;
  position.y += sin(angle) * distance;
  position = fminf(fmaxf(position, -BOUNDS), BOUNDS);
}
void SinewaveDropletCoordinate::backtrack()
{
  // Backtracks
  position.x = oldPosition.x;
  position.y = oldPosition.y;
}
StaticCoordinate SinewaveDropletCoordinate::getStaticCoord()
{
  float2 tempPos = position * scale;

  StaticCoordinate sc;
  sc.position.x = tempPos.x;// * scale;
  sc.position.z = tempPos.y;// * scale;
  // convert into 3D space
  float distance = wavelength * length(tempPos- ORIGIN);
  sc.position.y = sin(distance + SinewaveDropletCoordinate::time);
  sc.direction = TRUE_VERTICAL;

  //sc.direction.x = sin(distance);
  //sc.direction.y = cos(distance);
  //sc.direction.z = 1;

  // Horrible quick hack
  float2 inner = tempPos * 0.99f;
  float d2 = wavelength * length(inner - ORIGIN) + SinewaveDropletCoordinate::time;
  float3 inner3 = make_float3(inner.x, sin(d2) ,inner.y);
  float3 diff = normalize(inner3 - sc.position);
  float3 temp = normalize(cross(TRUE_VERTICAL, diff));
  float3 normal = normalize(cross(temp, diff));

  sc.direction = normal;

  return sc;
}
float SinewaveDropletCoordinate::getFastDistanceTo(NonPlanarCoordinate* other)
{
  SinewaveDropletCoordinate* sdc = (SinewaveDropletCoordinate*)other;
  return length(this->position - sdc->position);
}
float SinewaveDropletCoordinate::getDistanceTo(NonPlanarCoordinate* other)
{
  return getFastDistanceTo(other)*scale;
}

float SinewaveDropletCoordinate::scale = 1.0f;
float2 SinewaveDropletCoordinate::BOUNDS = make_float2(0.5f, 0.5f);
float2 SinewaveDropletCoordinate::ORIGIN = make_float2(0.0f);
float SinewaveDropletCoordinate::time = 0.0f;
