// A header file to contain everything related to the triangl emesh primitive
#ifndef TRIANGLE_MESH_PRIMITIVE_H
#define TRIANGLE_MESH_PRIMITIVE_H

#include <optix.h> // For CUDeviceptr
#include <cuda_runtime.h> // for float vectors
#include <sutil/Exception.h> // For CUDA_CHECK safeguards
#include <sutil/vec_math.h> // for vector maths
#include <algorithm> // for array manipulation between stack and heap.

#include <iostream>

using namespace std;

class TriangleMeshObject {
  public:
    
    // Constructor and destructor
    TriangleMeshObject();
    ~TriangleMeshObject();

    void setMeshDataToDefault();
    CUdeviceptr copyVerticesToDevice(); // Copy the vertices to the device
    CUdeviceptr getDeviceVertexPointer(); // Get a pointer to the device's vertices
    CUdeviceptr* getDeviceVertexPointerPointer(); // Convenience: Gets a pointer to the pointer to the device's vertices
    int getVertexCount(); // Gets the count of the vertices
    uint32_t getVertexCountUint(); // Convenience: Gets the vertex count as a uint32
    void deleteHostVertices();
    void deleteDeviceVertices();

    float3* vertices = nullptr;
    int vertexCount = 0;
    CUdeviceptr d_vertices=0;
  private:
};

#endif
