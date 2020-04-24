//#include "EyeGenerator.h"
#include "EquilibriumGenerator.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
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

  EquilibriumGenerator eqGen(100);
  eqGen.generateSphericalCoordinates();
  eqGen.stepSize = 0.01f;

  EyeGenerator* eg = (EyeGenerator*)&eqGen;
  cout << "Size of StaticCoordinate: " << sizeof(StaticCoordinate) << endl;

  //cout << "Size of BasicLight: " << sizeof(BasicLight) << endl;
  //eg.iterate();

  //EyeGenerator::basicIterator(&eg);

  cout << "Starting thread...";
  //thread testThread(EyeGenerator::basicIterator, &eg);
  thread testThread(EquilibriumGenerator::rieszSEnergyIterator, (EquilibriumGenerator*)eg);
  if(testThread.joinable())
    testThread.join();
  cout << "Thread joined." << endl;
  
  //eg.test();
  exit(EXIT_SUCCESS);
}
