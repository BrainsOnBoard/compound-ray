#include "TriangleMeshObject.h"

#define DEBUG

// Constructor and Destructor
TriangleMeshObject::TriangleMeshObject(){}
TriangleMeshObject::~TriangleMeshObject()
{
  deleteDeviceData();
  deleteHostData();
}

// Data assemblers
void TriangleMeshObject::setMeshDataToDefault()
{
  #ifdef DEBUG
  cout << "Adding default vertices to mesh..." << endl;
  #endif
  //// Verticies
  this->vertexCount = 4;
  float3 stack_vertices[this->vertexCount] = { 
                make_float3( -0.5f, -0.5f, 0.0f),
                make_float3(  0.5f, -0.5f, 0.0f),
                make_float3(  0.0f,  0.5f, 0.0f),
                make_float3(  0.0f,  -0.5f, -0.4f)
              };

  // Copy vertices from the stack into the heap:
  this->vertices = new float3[this->vertexCount];
  std::copy(stack_vertices, stack_vertices + this->vertexCount, &(this->vertices[0]));

  //// Vertex indicies
  this->triangleCount = 2;
  uint3 stack_triangles[this->triangleCount] = {
                make_uint3(0, 1, 2),
                make_uint3(0, 1, 3)
              };
  // Copy vertex indices to the heap:
  this->triangles = new uint3[this->triangleCount];
  std::copy(stack_triangles, stack_triangles + this->triangleCount, &(this->triangles[0]));
}
void TriangleMeshObject::performDataIntegrityCheck()
{
  if(d_vertices == 0 && d_triangles == 0 && triangleCount == 0 && vertexCount == 0)
    return;
  #ifdef DEBUG
  cout << "WARN: Mesh already populated. Depopulating..." << endl;
  #endif
  deleteHostData();
  deleteDeviceData();
}

// Getters
CUdeviceptr TriangleMeshObject::getDeviceVertexPointer()
{
  return d_vertices;
}
CUdeviceptr* TriangleMeshObject::getDeviceVertexPointerPointer()
{
  return &d_vertices;
}
unsigned int TriangleMeshObject::getVertexCount()
{
  return vertexCount;
}
CUdeviceptr TriangleMeshObject::getDeviceTrianglesPointer()
{
  return d_triangles;
}
CUdeviceptr* TriangleMeshObject::getDeviceTrianglesPointerPointer()
{
  return &d_triangles;
}
unsigned int TriangleMeshObject::getTriangleCount()
{
  return triangleCount;
}

// Device-side data handling
void TriangleMeshObject::copyDataToDevice()
{
  copyVerticesToDevice();
  copyTriangleIndiciesToDevice();
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
CUdeviceptr TriangleMeshObject::copyTriangleIndiciesToDevice()
{
  std::cout << "Copying triangle indicies to device (pointer @ "<< d_vertices << ")..." << std::endl;
  const size_t trianglesSize = sizeof(int3)*triangleCount;
  if(d_triangles == 0)
  {
    #ifdef DEBUG
    std::cout << "Device pointer is unassigned! Allocating memory on-device..." << std::endl;
    #endif
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_triangles ), trianglesSize) );
  }
  // Copy verticies to device memory
  CUDA_CHECK( cudaMemcpy(
              reinterpret_cast<void*>( d_triangles ),
              triangles,
              trianglesSize,
              cudaMemcpyHostToDevice
              ) );
  #ifdef DEBUG
  std::cout << "Memory allocated at: "<< d_triangles << std::endl;
  #endif
  return d_triangles;
}

void TriangleMeshObject::deleteHostData()
{
  #ifdef DEBUG
  cout<<"Freeing heap memory of vertices and triangles..."<<endl;
  #endif
  // Delete the vertices from the heap
  delete [] vertices;
  vertexCount = 0;
  // Delete the triangles from the heap
  delete [] triangles;
  triangleCount = 0;
}

void TriangleMeshObject::deleteDeviceData()
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
  // Delete the triangles from GPU if they've been assigned there.
  if(d_triangles != 0)
  {
    #ifdef DEBUG
    cout << "Freeing on-device triangle buffer at " << d_triangles << endl;
    #endif
    CUDA_CHECK( cudaFree((void*)d_triangles) );
    d_triangles= 0;// Reset the device pointer so everything knows that it's not pointing at anything.
  }
}
