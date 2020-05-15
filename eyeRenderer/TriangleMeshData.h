#ifndef TRIANGLE_MESH_DATA_H
#define TRIANGLE_MESH_DATA_H
#include "SbtRecord.h" // for general SbtRecord definition template

struct TriangleMeshData
{
  float3 colour;
}

typedef SbtRecord<TriangleMeshData> TriangleMeshSbtRecord;
#endif
