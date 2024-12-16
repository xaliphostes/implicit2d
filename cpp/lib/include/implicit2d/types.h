#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unordered_map>
#include <vector>

// Data point can be anywhere in the mesh
struct DataPoint {
    double x, y;             // position
    Eigen::Vector2d normal;  // normal to geological layer
    int containing_triangle; // ID of triangle containing this point
    Eigen::Vector3d
        barycentric; // barycentric coordinates in containing triangle
};

struct Vertex {
    double x, y;
    int id;
};

struct Triangle {
    int v1, v2, v3; // vertex indices
    int id;
};

struct Edge {
    int v1, v2;  // vertex indices
    bool is_cut; // true if edge represents a fault
    int id;
};

using Vertices = std::vector<Vertex>;
using Edges = std::vector<Edge>;
using Triangles = std::vector<Triangle>;
using DataPoints = std::vector<DataPoint>;
