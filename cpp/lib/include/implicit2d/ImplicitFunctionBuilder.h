#pragma once
#include "types.h"
#include <unordered_map>
#include <vector>

class ImplicitFunctionBuilder {
  public:
    ImplicitFunctionBuilder(
        const std::vector<Vertex> &vertices,
        const std::vector<Triangle> &triangles, const std::vector<Edge> &edges,
        const std::unordered_map<int, Eigen::Vector2d> &normals);

    DataPoint createDataPoint(double x, double y,
                              const Eigen::Vector2d &normal);
    void buildLinearSystem();
    Eigen::VectorXd solve();
    double evaluate(double x, double y);

  private:
    std::vector<std::vector<int>> buildEdgeConnectivity();

    bool pointInTriangle(double x, double y, const Triangle &tri);

    int findContainingTriangle(double x, double y);

    void computeBarycentricCoords(double x, double y, const Triangle &tri,
                                  double &lambda1, double &lambda2,
                                  double &lambda3);

    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;
    std::vector<Edge> edges;
    std::unordered_map<int, Eigen::Vector2d> normals; // sparse normal data

    // Matrix for the linear system
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
};
