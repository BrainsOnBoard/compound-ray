//  !! Note that this class can only store a maximum of `unsigned int` vertices and `unsigned int` triangles !!

// A header file to contain everything related to the triangl emesh primitive
#ifndef TRIANGLE_MESH_PRIMITIVE_H
#define TRIANGLE_MESH_PRIMITIVE_H

#include <optix.h> // For CUDeviceptr
#include <cuda_runtime.h> // for float vectors
#include <sutil/Exception.h> // For CUDA_CHECK safeguards
#include <sutil/vec_math.h> // for vector maths
#include <algorithm> // for array manipulation between stack and heap.
#include <vector>

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

class TriangleMeshObject {
  public:
    
    // Constructor and destructor
    TriangleMeshObject();
    ~TriangleMeshObject();

    void setMeshDataToDefault();
    void setMeshDataToPractice();
    void setMeshDataFromFile(const char* filename);

    // Device-side memory handling
    CUdeviceptr getDeviceVertexPointer(); // Get a pointer to the device's vertices
    CUdeviceptr* getDeviceVertexPointerPointer(); // Convenience: Gets a pointer to the pointer to the device's vertices
    CUdeviceptr getDeviceTrianglesPointer(); // Get a pointer to the device's triangle buffer
    CUdeviceptr* getDeviceTrianglesPointerPointer(); // Convenience
    void copyDataToDevice(); // Copies accross the vertices and triangles to storage on-device

    unsigned int getVertexCount() const; // Gets the count of the vertices
    unsigned int getTriangleCount() const;// Gets the count of the triangles

    void deleteHostData(); // Delete all verticies and triangles host-side
    void deleteDeviceData(); // Delete all verticies and triangles device-side

  private:
    // Static helper functions
    static const vector<string> splitString(const string& s, const string& delimiter);
    static void populateAndAddTriplet(const vector<string>& pointData, vector<uint3>& triangles, const size_t* orderArray);
    static void populateAndAddTriplet(const vector<string>& pointData, vector<uint3>& triangles);
    static const uint3 UNIT_UINT3_CUBE;
    static const size_t STANDARD_INDEX_ORDER[3];
    static const size_t OFF_RECT_INDEX_ORDER[3];
    float3* vertices = nullptr;

    // Dynamic functions and variables
    unsigned int vertexCount = 0;
    CUdeviceptr d_vertices=0;
    
    uint3* triangles = nullptr;// A list of indicies of triangles (in sets of 3)
    unsigned int triangleCount = 0;
    CUdeviceptr d_triangles=0;

    CUdeviceptr copyVerticesToDevice(); // Copy the vertices to the device
    CUdeviceptr copyTriangleIndiciesToDevice(); // Copy the vertex indicies to the device

    void performDataIntegrityCheck();
};

#endif
