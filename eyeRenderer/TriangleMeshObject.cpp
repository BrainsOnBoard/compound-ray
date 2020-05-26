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
void TriangleMeshObject::setMeshDataFromFile(const char* filename)
{
  cout<<"Opening file \""<<filename<<"\"..."<<endl;
  performDataIntegrityCheck();
  string line;
  ifstream filestream(filename);
  if(filestream.is_open())
  {
    // Read in the file
    unsigned int vc = 0; // The number of vertices in the file.
    vector<float3> vs; // The vertices of the file
    unsigned int tc = 0; // The number of trianlges in the file.
    vector<uint3> tris;
    while(getline(filestream, line))
    {
      vector<string> lineData = splitString(line, " ");
      if(lineData.front().compare("v") == 0)
      {
        float3 vert = make_float3(stof(lineData[1]), stof(lineData[2]), stof(lineData[3]));
        vs.push_back(vert);
        cout<<"Made vert: " << vert.x << ", " << vert.y << ", " << vert.z << endl;
        vc++;
      }
      if(lineData.front().compare("f") == 0)
      {
        uint3 indices = make_uint3(stoi(splitString(lineData[1], "/").front()), stoi(splitString(lineData[2], "/").front()), stoi(splitString(lineData[3], "/").front()));
        indices = indices - UNIT_UINT3_CUBE;
        tris.push_back(indices);
        cout<<"Made face: " << indices.x << ", " << indices.y << ", " << indices.z << endl;
        tc++;
      }
    }
    cout<<"Parsed "<<vc<<" vertices and "<<tc<<" triangles."<<endl;
    filestream.close();

    // Copy the data to this object's heap
    this->vertexCount = vc;
    this->triangleCount = tc;
    this->vertices = new float3[this->vertexCount];
    this->triangles = new uint3[this->triangleCount];
    copy(vs.data(), vs.data() + this->vertexCount, &(this->vertices[0]));
    copy(tris.data(), tris.data() + this->triangleCount, &(this->triangles[0]));

  }else{
    std::cout << "ERROR: Unable to open \"" << filename << "\"";
  }

  cout<<"FIRST VERTEX DATA: (" << this->vertices[0].x << ", " << this->vertices[0].y << ", " << this->vertices[0].z << ")" <<endl;
}

void TriangleMeshObject::setMeshDataToDefault()
{
  #ifdef DEBUG
  cout << "Adding default vertices to mesh..." << endl;
  #endif
  performDataIntegrityCheck();
  //// Verticies
  this->vertexCount = 4;
  float3 stack_vertices[this->vertexCount] = { 
                make_float3( -0.5f, -0.5f, 0.0f),
                make_float3(  0.5f, -0.5f, 0.0f),
                make_float3(  0.0f,  0.5f, 0.0f),
                make_float3(  0.0f,  -0.5f, 0.4f)
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
  std::copy(stack_triangles, stack_triangles + (this->triangleCount), &(this->triangles[0]));
}
void TriangleMeshObject::setMeshDataToPractice()
{
  #ifdef DEBUG
  cout << "Adding practice vertices to mesh..." << endl;
  #endif
  performDataIntegrityCheck();
  //// Verticies
  this->vertexCount = 8;
  float3 stack_vertices[this->vertexCount] = { 
                make_float3( 1.0f, 3.36344f, -1.0f),
                make_float3( 1.0f, 1.36344f, -1.0f),
                make_float3( 1.0f, 3.36344f,  1.0f),
                make_float3( 1.0f, 1.36344f,  1.0f),
                make_float3(-1.0f, 3.36344f, -1.0f),
                make_float3(-1.0f, 1.36344f, -1.0f),
                make_float3(-1.0f, 3.36344f,  1.0f),
                make_float3(-1.0f, 1.36344f,  1.0f)
              };

  // Copy vertices from the stack into the heap:
  this->vertices = new float3[this->vertexCount];
  std::copy(stack_vertices, stack_vertices + this->vertexCount, &(this->vertices[0]));

  //// Vertex indicies
  this->triangleCount = 6;
  uint3 stack_triangles[this->triangleCount] = {
                make_uint3(0, 4, 6),
                make_uint3(3, 2, 6),
                make_uint3(7, 6, 4),
                make_uint3(5, 1, 3),
                make_uint3(1, 0, 2),
                make_uint3(5, 4, 0)
              };
  // Copy vertex indices to the heap:
  this->triangles = new uint3[this->triangleCount];
  std::copy(stack_triangles, stack_triangles + this->triangleCount, &(this->triangles[0]));
}
// Stops memory leaks and collisions by deleting everything
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
unsigned int TriangleMeshObject::getVertexCount() const
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
unsigned int TriangleMeshObject::getTriangleCount() const
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
  const size_t verticesSize = sizeof(this->vertices[0])*vertexCount;
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
  std::cout << "Copying triangle indicies to device (pointer @ "<< d_triangles << ")..." << std::endl;
  const size_t trianglesSize = sizeof(this->triangles[0])*triangleCount;
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
  vertices = nullptr;// Reset pointer
  vertexCount = 0;
  // Delete the triangles from the heap
  delete [] triangles;
  triangles = nullptr;// Reset pointer
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

//// Static helper messages:
const vector<string> TriangleMeshObject::splitString(const string& s, const string& deliminator)
{
  vector<string> output;
  const size_t delimSize = deliminator.size();
  size_t lastDelimLoc = 0;
  size_t delimLoc = s.find(deliminator, 0);
  while(delimLoc != std::string::npos)
  {
    if(delimLoc != lastDelimLoc)
      output.push_back(s.substr(lastDelimLoc, delimLoc-lastDelimLoc));
    lastDelimLoc = delimLoc + delimSize;
    delimLoc = s.find(deliminator, lastDelimLoc);
  }
  // Push either the whole thing if it's not found, or the last segment if there were deliminators
  output.push_back(s.substr(lastDelimLoc, s.size()));
  return output;
}

const uint3 TriangleMeshObject::UNIT_UINT3_CUBE = make_uint3(1,1,1);
