// The bit that actually loads the surfaces.
#pragma once

#include <string>
#include <set>
#include <map>
#include <utility>


class Surface
{
  public:
    bool load(const std::string & filepath); // Loads a surface
    void precomputeInterVertexDistances();
    void clear(); // Clears the surfaces from this object

    void printVertexGraph();

  private:
    struct Vertex
    {
      float x,y,z; // Position
      std::set<size_t> connectedVertices; // The indices of connected vertices
    };

    typedef std::pair<size_t, Vertex> indexVertexPair;
    std::map<size_t, Vertex> vertexGraph;
};
