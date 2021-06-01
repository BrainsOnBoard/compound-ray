#include "Surface.h"

#include <iostream>

// Include tinyObjLoader
//#define TINYOBJLOADER_USE_DOUBLE // Enable double-precision object loading
#define TINYOBJLOADER_IMPLEMENTATION
#include <support/tinyobjloader/tiny_obj_loader.h>


using namespace std;

bool Surface::load (const std::string & filepath)
{
  // Load the surface using tiny OBJ loader ( https://github.com/tinyobjloader/tinyobjloader )
  cout << "Loading surface from file " << filepath << "..." << endl;

  tinyobj::ObjReaderConfig readerConfig;
  readerConfig.mtl_search_path = "./";

  tinyobj::ObjReader reader;

  // Attempt to load the file
  if(!reader.ParseFromFile(filepath, readerConfig))
  {
    if(!reader.Error().empty())
    {
      cerr << "[Error] Reading obj file using TinyObjReader: " << reader.Error();
    }
    return false;
  }

  // Throw a warning if loader raised one
  if(!reader.Warning().empty())
  {
    cout << "[Warning] When reading obj file using TinyObjReader: " << reader.Warning();
  }

  auto& attrib = reader.GetAttrib();
  auto& shapes = reader.GetShapes();

  if(shapes.size() > 1)
  {
    cout << "[Warning] This file contains more than one mesh, only the first will be used." << endl;
  }
  auto& shape = shapes[0];

  cout << "Obj file has " << attrib.vertices.size()/3 << " vertices..." << endl;

  // Loop over the polygons, add their vertices to the vertex graph.
  size_t indexOffset = 0;
  for(size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
  {
    size_t fv = size_t(shape.mesh.num_face_vertices[f]);

    // Loop over vertices in the face
    for(size_t v = 0; v < fv; v++)
    {
      // Create a new vertex for this in the vertex graph
      size_t idx = size_t(shape.mesh.indices[indexOffset + v].vertex_index);// Get attrib index of this face

      // First check if this vertex has already been listed
      Vertex* vert = nullptr;
      auto it = vertexGraph.find(idx);
      if(it == vertexGraph.end())
      {
        // It hasn't been listed, so just insert it
        Vertex newVert;
        newVert.x = float(attrib.vertices[3*idx+0]);
        newVert.y = float(attrib.vertices[3*idx+1]);
        newVert.z = float(attrib.vertices[3*idx+2]);
        //newVert.connectedVertices = new std::set<tinyobj::index_t>();
        vertexGraph.insert(indexVertexPair(idx, newVert));
        vert = &(vertexGraph.find(idx)->second);
      }
      else
      {
        // It has already been listed, so just pull a reference to it
        vert = &(it->second);
      }

      // Now attach the connected vertices from this polygon.
      for(size_t otherVert = 0; otherVert < fv; otherVert++)
      {
        if(otherVert == v)
          continue; // Don't connect this vert to itself

        // Otherwise, connect the other vert to this vert
        vert->connectedVertices.insert(size_t(shape.mesh.indices[indexOffset + otherVert].vertex_index));
      }
    }

    indexOffset += fv;
  }

  cout << "Vertex graph constructed (" << vertexGraph.size() << " unique vertices)." << endl;

  return true; // No errors occured.
}

void Surface::precomputeInterVertexDistances()
{
}
void Surface::printVertexGraph()
{
  for (indexVertexPair currentIndexVertexPair : vertexGraph)
  {
    size_t idx = currentIndexVertexPair.first;
    Vertex v = currentIndexVertexPair.second;
    cout << "Vertex at index " << idx << ": " << endl;
    cout << "\tvertex position: ("<< v.x <<","<< v.y << "," << v.z << ")" << endl;
    cout << "\t" << v.connectedVertices.size() << " vertex connections: ";
    //for( std::set<size_t>::iterator it = v.connectedVertices.begin(); it!=v.connectedVertices.end(); ++it)
    for (auto connectedIndex : v.connectedVertices)
    {
      cout << connectedIndex << " ";
    }
    cout << endl;
  }
}

void Surface::clear()
{
//  while(!vertexGraph.empty())
//  {
//    // Destroy all elements of the map safely
//    // Probably by calling some destruction function on the vertex, then calling erase
//  }
}
