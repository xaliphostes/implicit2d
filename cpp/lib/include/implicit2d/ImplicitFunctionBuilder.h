#pragma once
#include "types.h"
#include <unordered_map>
#include <vector>

class ImplicitFunctionBuilder {
  public:
    ImplicitFunctionBuilder();
    ImplicitFunctionBuilder(const std::vector<Vertex> &vertices,
                            const std::vector<Triangle> &triangles,
                            const std::vector<Edge> &edges,
                            const std::vector<DataPoint> &data_points);

    void beginDescription();
    void addVertex(double, double);
    void addVertices(const std::vector<double> &);
    void addTriangle(uint32_t, uint32_t, uint32_t);
    void addTriangles(const std::vector<uint32_t> &);
    void addEdge(uint32_t, uint32_t);
    void addDataPoint(double, double, const Normal &);
    void endDescription();

    void setDataPoints(const std::vector<DataPoint> &new_data_points);

    DataPoint createDataPoint(double x, double y, const Normal &normal);
    void buildLinearSystem();
    const Eigen::VectorXd &solve();
    double evaluate(double x, double y) const;

    void visualize(
        const std::string &filename,
        const VisualizationOptions &options = VisualizationOptions()) const;

    void visualizeColorMap(const std::string &filename,
                           int nx = 100, // number of samples in x
                           int ny = 100, // number of samples in y
                           bool showMesh = true) const;

  private:
    std::vector<std::vector<std::pair<double, double>>>
    computeContours(double min_val, double max_val, int num_contours) const;

    std::vector<std::vector<uint32_t>> buildEdgeConnectivity();
    void precomputeTriangleBasis();

    bool pointInTriangle(double x, double y, const Triangle &tri) const;

    int findContainingTriangle(double x, double y) const;

    void computeBarycentricCoords(double x, double y, const Triangle &tri,
                                  double &lambda1, double &lambda2,
                                  double &lambda3) const;

    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;
    std::vector<Edge> edges;
    std::vector<DataPoint> data_points; // data points can be anywhere in mesh
    std::vector<Eigen::Matrix<double, 2, 3>> triangle_basis_gradients;

    // Matrix for the linear system
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
    Eigen::VectorXd solution_;
};
