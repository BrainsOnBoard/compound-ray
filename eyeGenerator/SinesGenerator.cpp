#include "SinesGenerator.h"

SinesGenerator::SinesGenerator(int coordinateCount)
{
  separation = 1.0f;
  this->coordinateCount = coordinateCount;
  this->size = (int)sqrt(coordinateCount);
  coordinates = (StaticCoordinate*) calloc(coordinateCount, sizeof(StaticCoordinate));
  if(coordinates == NULL)
  {
    std::cout << "ERROR: Insufficient memory to store sines ommatidial coordinates." << std::endl;
    exit(0);
  }
  for(int i = 0; i<size; i++)
  {
    for(int o = 0; o<size; o++)
    {
      coordinates[i*size + o].position.x = i *separation;
      coordinates[i*size + o].position.y = o *separation;
      coordinates[i*size + o].position.z = 0.0f;
    }
  }
  stopFlag = false;
  newDataReadyFlag = true;
}
SinesGenerator::~SinesGenerator()
{
  //for(int i = 0; i<gridSize; i++)
  //  for(int o = 0; o<gridSize; o++)
  //  {
  //    delete coordinates[i*gridSize + o];
  //  }
  free(coordinates);
}

StaticCoordinate SinesGenerator::getCoordinateInfo(int i)
{
  StaticCoordinate sc;
  sc.position.x = coordinates[i].position.x;
  sc.position.y = coordinates[i].position.y;
  sc.position.z = coordinates[i].position.z;
  //cout<< coordinates[0].position.x<<endl;
  //sc.position = make_float3(1.0f);
  return sc;
}
void SinesGenerator::stop()
{
  stopFlag = true;
}

bool SinesGenerator::hasNewDataReady()
{
  if(newDataReadyFlag)
  {
    newDataReadyFlag = false;
    return true;
  }
  return false;
}

// Udulates the positions and orientations of the array in a separate threead
void SinesGenerator::sinesUdulator(SinesGenerator* sg)
{
  int gridSize = sg->size;
  cout << "Running sine udulation...";
  while(!sg->stopFlag)
  {
    for(int i = 0; i<gridSize; i++)
      for(int o = 0; o<gridSize; o++)
      {
        //StaticCoordinate coord = sg->coordinates[i*gridSize + o];
        //cout << coord.position.x << endl;
        //coord.position.y = sin(10.0f);
      }
    sg->newDataReadyFlag = true;
    this_thread::sleep_for(chrono::milliseconds(100));
  }
  cout << "Sines have stopped udulating." << endl;
}
