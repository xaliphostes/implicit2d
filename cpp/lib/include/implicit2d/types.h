#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unordered_map>
#include <vector>

using Normal = Eigen::Vector2d;
using Barycentric = Eigen::Vector3d;

// Data point can be anywhere in the mesh
struct DataPoint {
    double x, y;
    Normal normal;
    uint32_t containing_triangle; // ID of triangle containing this point
    Barycentric barycentric;
};

struct Vertex {
    double x, y;
    uint32_t id;
};

struct Triangle {
    uint32_t v1, v2, v3;
    uint32_t id;
};

struct Edge {
    uint32_t v1, v2;
    bool is_cut; // true if edge represents a fault
    uint32_t id;
};

using Vertices = std::vector<Vertex>;
using Edges = std::vector<Edge>;
using Triangles = std::vector<Triangle>;
using DataPoints = std::vector<DataPoint>;

struct VisualizationOptions {
    uint nbIsos{50};
    bool showBBox{true};
    bool showContours{true};
    bool showMesh{true};
    bool showDataPoints{true};
};