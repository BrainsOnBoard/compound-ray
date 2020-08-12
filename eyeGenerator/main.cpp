//#include "EyeGenerator.h"
#include "EquilibriumGenerator.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <thread>

#include "NonPlanarCoordinate.h"

using namespace std;

void threadFunc()
{
  cout << "Thread made." << endl;
}
int main(int argc, char** argv)
{
  srand(42);

  //EquilibriumGenerator eqGen(100);
  //eqGen.generateSphericalCoordinates();
  //eqGen.stepSize = 0.01f;

  //EyeGenerator* eg = (EyeGenerator*)&eqGen;
  //cout << "Size of StaticCoordinate: " << sizeof(StaticCoordinate) << endl;

  ////cout << "Size of BasicLight: " << sizeof(BasicLight) << endl;
  ////eg.iterate();

  ////EquilibriumGenerator::rieszSEnergyIterator((EquilibriumGenerator*)eg);

  //cout << "Starting thread...";
  //thread testThread(EquilibriumGenerator::rieszSEnergyIterator, (EquilibriumGenerator*)eg);
  //if(testThread.joinable())
  //  testThread.join();
  //cout << "Thread joined." << endl;

  //////// SINEWAVE DROPLET GENERATOR:
  //EquilibriumGenerator eqsGen(100);
  //eqsGen.generateSinewaveDropletCoordinates();
  //eqsGen.stepSize = 0.01f;
  //EyeGenerator* egs = (EyeGenerator*)&eqsGen;

  //// Note that the sinewave droplet would be animated externally in here:
  //cout << "Starting thread...";
  //thread testThreadTwo(EquilibriumGenerator::rieszSEnergyIterator, (EquilibriumGenerator*)eg);
  //if(testThreadTwo.joinable())
  //  testThreadTwo.join();
  //cout << "Thread joined." << endl;


  // Generate a basic eye
  size_t ommCount = 1000;
  EquilibriumGenerator basicEye(ommCount);
  basicEye.generateSphericalCoordinates();
  basicEye.stepSize = 0.0001f;
  //basicEye.varianceCap = 0.05f;
  basicEye.varianceCap = 1.10f;
  //basicEye.varianceCap = 2.5f;

  EyeGenerator* eg2 = (EyeGenerator*)&basicEye;
  thread generatorThread(EquilibriumGenerator::rieszSEnergyIterator, (EquilibriumGenerator*)eg2);
  if(generatorThread.joinable())
    generatorThread.join();

  ofstream output;
  output.open("test1000-horizontallyAcute.eye");
  for(size_t i = 0; i<ommCount; i++)
  {
    StaticCoordinate sc = eg2->getCoordinateInfo(i);
    output << sc.position.x<<" "<<sc.position.y<<" "<<sc.position.z<<" "<<sc.direction.x<<" "<<sc.direction.y<<" "<<sc.direction.z<<" 1.0\n";
    cout<<"Ommatidium: ("<<sc.direction.x<<", "<<sc.direction.y<<", "<<sc.direction.z<<")\n";
  }
  output.close();
  
  exit(EXIT_SUCCESS);
}
