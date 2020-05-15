#include "TriangleMeshObject.h"

#define DEBUG

// Constructor
TriangleMeshObject::TriangleMeshObject(){}
TriangleMeshObject::~TriangleMeshObject()
{
  deleteHostVertices();
  deleteDeviceVertices();
}

void TriangleMeshObject::setMeshDataToDefault()
{
  #ifdef DEBUG
  std::cout << "Adding default vertices to mesh..." << endl;
  #endif
  this->vertexCount = 6;
  float3 stack_vertices[this->vertexCount] = { 
                make_float3( -0.5f, -0.5f, 0.0f),
                make_float3(  0.5f, -0.5f, 0.0f),
                make_float3(  0.0f,  0.5f, 0.0f),
                make_float3( -1.5f, -0.5f, 0.0f),
                make_float3(  0.5f, -0.5f, 0.0f),
                make_float3(  0.0f,  -0.5f, 0.0f)
              };

  this->vertices = new float3[this->vertexCount];

  // Copy from the stack into the heap:
  std::copy(stack_vertices, stack_vertices + this->vertexCount, &(this->vertices[0]));
}

CUdeviceptr TriangleMeshObject::copyVerticesToDevice()
{
  std::cout << "Copying vertices to device (pointer @ "<< d_vertices << ")..." << std::endl;
  const size_t verticesSize = sizeof(float3)*vertexCount;
  if(d_vertices == 0)
  {
    #ifdef DEBUG
    std::cout << "Device pointer is unassigned! Allocating memory on-device..." << std::endl;
    #endif
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_vertices ), verticesSize) );
  }
  // Copy verticies to device memory
  CUDA_CHECK( cudaMemcpy(
              reinterpret_cast<void*>( d_vertices ),
              vertices,
              verticesSize,
              cudaMemcpyHostToDevice
              ) );
  #ifdef DEBUG
  std::cout << "Memory allocated at: "<< d_vertices << std::endl;
  #endif
  return d_vertices;
}
CUdeviceptr TriangleMeshObject::getDeviceVertexPointer()
{
  return d_vertices;
}
CUdeviceptr* TriangleMeshObject::getDeviceVertexPointerPointer()
{
  return &d_vertices;
}

int TriangleMeshObject::getVertexCount()
{
  return vertexCount;
}
uint32_t TriangleMeshObject::getVertexCountUint()
{
  return static_cast<uint32_t>(getVertexCount());
}
void TriangleMeshObject::deleteHostVertices()
{
  #ifdef DEBUG
  cout<<"Freeing heap memory of vertices..."<<endl;
  #endif
  // Delete the vertices from the heap
  delete [] vertices;
  vertexCount = 0;
}
void TriangleMeshObject::deleteDeviceVertices()
{
  // Delete the vertices from the GPU if they've been assigned there.
  if(d_vertices != 0)
  {
    #ifdef DEBUG
    cout << "Freeing on-device vertex buffer at " << d_vertices << endl;
    #endif
    CUDA_CHECK( cudaFree((void*)d_vertices) );
    d_vertices = 0;// Reset the device pointer so everything knows that it's not pointing at anything.
  }
}
