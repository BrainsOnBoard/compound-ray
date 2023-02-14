#pragma once

#include <sutil/sutilapi.h>
#include <sutil/vec_math.h>
#include <sutil/Matrix.h>
#include <sutil/Aabb.h>
#include <vector>
#include <string>

#include "hitscanprocessingapi.h"

//#define TINYGLTF_IMPLEMENTATION
//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined( WIN32 )
#pragma warning( push )
#pragma warning( disable : 4267 )
#endif
#include <support/tinygltf/tiny_gltf.h>
#if defined( WIN32 )
#pragma warning( pop )
#endif

namespace sutil {
namespace hitscan {

struct Triangle {
  float3 p0, p1, p2;
};

struct TriangleMesh{
  std::string name;
  Aabb worldAabb;
  Aabb objectAabb;
  Matrix<4,4> transform;
  std::vector<Triangle> triangles;

  // Prints this triangle mesh's information
  void print();
};

// Performs hitscan stuff
HITSCANAPI const bool isPointWithinMesh(TriangleMesh& tm, float3 worldPoint);
HITSCANAPI const unsigned int countMeshRayIntersections(TriangleMesh& tm, float3 rayStart, float3 rayDir, float limit = 100000.0f, bool debug = false);
HITSCANAPI void calculateObjectAabb(TriangleMesh& tm);
HITSCANAPI void calculateWorldAabbUsingTransformAndObjectAabb(TriangleMesh& tm);

// Populates a TriangleMesh with a tinyGlTF primitive
HITSCANAPI void populateTriangleMesh(TriangleMesh& tm, const tinygltf::Mesh& mesh, const tinygltf::Model& model);
template <typename IndexType>
HITSCANAPI void getTriangles(TriangleMesh& tm, const tinygltf::Model& model, const tinygltf::Primitive& primitive);
template <typename IndexType, typename FloatType>
HITSCANAPI void getTrianglesInFloatForm(TriangleMesh& tm, const tinygltf::Model& model, const tinygltf::Primitive& primitive);

}
}
