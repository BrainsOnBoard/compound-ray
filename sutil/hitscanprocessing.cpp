#include "hitscanprocessing.h"
#include <iostream>

//#define TINYGLTF_IMPLEMENTATION
//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#if defined( WIN32 )
//#pragma warning( push )
//#pragma warning( disable : 4267 )
//#endif
//#include <support/tinygltf/tiny_gltf.h>
//#if defined( WIN32 )
//#pragma warning( pop )
//#endif

/////////////////////////////////////////////////////////////
// Performing hitscans
/////////////////////////////////////////////////////////////

const bool sutil::hitscan::isPointWithinMesh(sutil::hitscan::TriangleMesh& tm, float3 worldPoint)
{
  float3 objectPos = make_float3(tm.transform.inverse() * make_float4(worldPoint));
  float3 rayStartPos = objectPos;
  rayStartPos.x = tm.objectAabb.m_min.x - 1.0f;
  float3 rayDir = normalize(objectPos - rayStartPos);
  float d = 0.12f;
  unsigned int intersectionCount = 0;
  for(Triangle triangle : tm.triangles )
  {
    // Basically just return true if near a vertex, forming spheres around the vertices for testing purposes
    //if(length(triangle.p0 - objectPos) < d ||
    //   length(triangle.p1 - objectPos) < d ||
    //   length(triangle.p2 - objectPos) < d )
    //   return true;

    //// Test if the triangle intersects the ray
    float3 planeNormal = normalize(cross((triangle.p1-triangle.p0), (triangle.p2-triangle.p0)));

    float denominator = dot(planeNormal, rayDir);
    if(denominator == 0)
      continue; // Don't test against this triangle if it's parallel to the ray (infinite intersections)

    float distanceToPlaneAlongRay = dot((triangle.p0 - rayStartPos), planeNormal) / denominator;

    if(distanceToPlaneAlongRay == 0)
      continue; // The ray sits in the plane, so don't count it.

    float3 hitLocation = rayStartPos + distanceToPlaneAlongRay * rayDir;

    // Skip this ray if it intersected behind the raycast direction
    // or if the intersection location was past where the target location is, in object-space.
    if(distanceToPlaneAlongRay < 0 || hitLocation.x > objectPos.x)
      continue;
    
    //// Make sure that the intersection is less than or equal to limit
    float3 edge, fromEdgeStart, crossProd;

    //// First edge
    edge = triangle.p1 - triangle.p0;
    fromEdgeStart = hitLocation - triangle.p0;
    crossProd = cross(edge, fromEdgeStart);
    if(dot(planeNormal, crossProd) < 0)
      continue;

    //// Second edge
    edge = triangle.p2 - triangle.p1;
    fromEdgeStart = hitLocation - triangle.p1;
    crossProd = cross(edge, fromEdgeStart);
    if(dot(planeNormal, crossProd) < 0)
      continue;

    //// Third edge
    edge = triangle.p0 - triangle.p2;
    fromEdgeStart = hitLocation - triangle.p2;
    crossProd = cross(edge, fromEdgeStart);
    if(dot(planeNormal, crossProd) < 0)
      continue;
    
    // Finally, if it's a good hit and inside the triangle, then add to the intersection count
    intersectionCount++;
  }
  return intersectionCount%2 == 1;
}

void sutil::hitscan::calculateObjectAabb(sutil::hitscan::TriangleMesh& tm)
{
  float3 minPos = tm.triangles[0].p0;
  float3 maxPos = tm.triangles[0].p0;

  for(auto triangle : tm.triangles)
  {
    minPos = fminf(minPos, triangle.p0);
    minPos = fminf(minPos, triangle.p1);
    minPos = fminf(minPos, triangle.p2);
    maxPos = fmaxf(maxPos, triangle.p0);
    maxPos = fmaxf(maxPos, triangle.p1);
    maxPos = fmaxf(maxPos, triangle.p2);
  }

  tm.objectAabb = Aabb(minPos, maxPos);
}

void sutil::hitscan::calculateWorldAabbUsingTransformAndObjectAabb(sutil::hitscan::TriangleMesh& tm)
{
  tm.worldAabb = tm.objectAabb;
  tm.worldAabb.transform(tm.transform);
}



/////////////////////////////////////////////////////////////
// Loading gltf models as triangle meshes
/////////////////////////////////////////////////////////////

void sutil::hitscan::populateTriangleMesh(sutil::hitscan::TriangleMesh& tm, const tinygltf::Mesh& mesh, const tinygltf::Model& model)
{
  for( auto& primitive : mesh.primitives )
  {
      if( primitive.mode != TINYGLTF_MODE_TRIANGLES ) // Ignore non-triangle meshes
      {
          std::cerr << "\tNon-triangle primitive: skipping\n";
          continue;
      }

      // Switch based on how the indicies are stored
      const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
      switch(indexAccessor.componentType)
      {
        default:
        case TINYGLTF_COMPONENT_TYPE_BYTE:
          getTriangles<int8_t>(tm, model, primitive);
          break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
          getTriangles<uint8_t>(tm, model, primitive);
          break;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
          getTriangles<int16_t>(tm, model, primitive);
          break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
          getTriangles<uint16_t>(tm, model, primitive);
          break;
        case TINYGLTF_COMPONENT_TYPE_INT:
          getTriangles<int32_t>(tm, model, primitive);
          break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
          getTriangles<uint32_t>(tm, model, primitive);
          break;
     }
  }
}

template <typename IndexBufferType>
void sutil::hitscan::getTriangles(sutil::hitscan::TriangleMesh& tm, const tinygltf::Model& model, const tinygltf::Primitive& primitive)
{
  // Switch based on how the position data is stored
  const tinygltf::Accessor& positionAccessor = model.accessors[primitive.attributes.at("POSITION")];
  switch(positionAccessor.componentType)
  {
    default:
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
      getTrianglesInFloatForm<IndexBufferType, float>(tm, model, primitive);
      break;
    case TINYGLTF_COMPONENT_TYPE_DOUBLE:
      getTrianglesInFloatForm<IndexBufferType, double>(tm, model, primitive);
      break;
  }
}

template <typename IndexBufferType, typename PositionBufferType>
void sutil::hitscan::getTrianglesInFloatForm(sutil::hitscan::TriangleMesh& tm, const tinygltf::Model& model, const tinygltf::Primitive& primitive)
{
  // Actually get the triangles into the TriangleMesh (surprisingly simple)
  const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
  const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
  const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

  const tinygltf::Accessor& positionAccessor = model.accessors[primitive.attributes.at("POSITION")];
  const tinygltf::BufferView& positionBufferView = model.bufferViews[positionAccessor.bufferView];
  const tinygltf::Buffer& positionBuffer = model.buffers[positionBufferView.buffer];

  const IndexBufferType* indices = reinterpret_cast<const IndexBufferType*>(&indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);
  const PositionBufferType* positions = reinterpret_cast<const PositionBufferType*>(&positionBuffer.data[positionBufferView.byteOffset + positionAccessor.byteOffset]);

  for(int i = 0; i<indexAccessor.count; i+=3)
  {
    tm.triangles.push_back({make_float3(positions[indices[ i ]*3],positions[indices[ i ]*3+1],positions[indices[ i ]*3+2]),
                            make_float3(positions[indices[i+1]*3],positions[indices[i+1]*3+1],positions[indices[i+1]*3+2]),
                            make_float3(positions[indices[i+2]*3],positions[indices[i+2]*3+1],positions[indices[i+2]*3+2])
                           });
  }
}

void sutil::hitscan::TriangleMesh::print()
{
  for(auto tri : triangles)
  {
    std::cout << tri.p0.x << "," << tri.p0.y << "," << tri.p0.z << "|"
              << tri.p1.x << "," << tri.p1.y << "," << tri.p1.z << "|"
              << tri.p2.x << "," << tri.p2.y << "," << tri.p2.z << ":";
  }
  std::cout << "\n";
}
