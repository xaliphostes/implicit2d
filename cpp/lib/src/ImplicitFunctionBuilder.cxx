#include <implicit2d/ImplicitFunctionBuilder.h>

ImplicitFunctionBuilder::ImplicitFunctionBuilder(
    const std::vector<Vertex> &vertices, const std::vector<Triangle> &triangles,
    const std::vector<Edge> &edges,
    const std::unordered_map<int, Eigen::Vector2d> &normals)
    : vertices(vertices), triangles(triangles), edges(edges), normals(normals) {
}

// Build the linear system that will solve for the implicit function
void ImplicitFunctionBuilder::buildLinearSystem() {
    int n = vertices.size();
    A.resize(n, n);
    b.resize(n);
    b.setZero();

    // Get edge connectivity respecting cuts
    auto edge_neighbors = buildEdgeConnectivity();

    // Build sparse matrix structure
    std::vector<Eigen::Triplet<double>> triplets;

    // For each vertex
    for (int i = 0; i < n; i++) {
        double sum_weights = 0.0;
        const auto &neighbors = edge_neighbors[i];

        // If we have normal data at this vertex
        if (normals.count(i) > 0) {
            // Add gradient constraint
            const Eigen::Vector2d &normal = normals.at(i);
            for (int j : neighbors) {
                Eigen::Vector2d edge_vector(vertices[j].x - vertices[i].x,
                                            vertices[j].y - vertices[i].y);
                double weight = edge_vector.dot(normal);
                triplets.emplace_back(i, j, weight);
                sum_weights += std::abs(weight);
            }
        }
        // Otherwise use Laplacian smoothing
        else {
            for (int j : neighbors) {
                triplets.emplace_back(i, j, 1.0);
                sum_weights += 1.0;
            }
        }

        // Add diagonal element
        triplets.emplace_back(i, i, -sum_weights);
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
}

// Solve the linear system to get the implicit function values
Eigen::VectorXd ImplicitFunctionBuilder::solve() {
    // Use a sparse solver (BiCGSTAB with diagonal preconditioner)
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    Eigen::VectorXd x = solver.solve(b);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to solve linear system");
    }

    return x;
}

// Evaluate the implicit function at any point using linear interpolation
double ImplicitFunctionBuilder::evaluate(double x, double y) {
    // Find containing triangle using point location
    for (const Triangle &tri : triangles) {
        if (pointInTriangle(x, y, tri)) {
            // Compute barycentric coordinates
            double lambda1, lambda2, lambda3;
            computeBarycentricCoords(x, y, tri, lambda1, lambda2, lambda3);

            // Interpolate function value
            Eigen::VectorXd sol = solve();
            return lambda1 * sol[tri.v1] + lambda2 * sol[tri.v2] +
                   lambda3 * sol[tri.v3];
        }
    }
    throw std::runtime_error("Point not found in mesh");
}

// Helper function to build edge connectivity
std::vector<std::vector<int>> ImplicitFunctionBuilder::buildEdgeConnectivity() {
    std::vector<std::vector<int>> edge_neighbors(vertices.size());
    for (const Edge &edge : edges) {
        if (!edge.is_cut) {
            edge_neighbors[edge.v1].push_back(edge.v2);
            edge_neighbors[edge.v2].push_back(edge.v1);
        }
    }
    return edge_neighbors;
}

bool ImplicitFunctionBuilder::pointInTriangle(double x, double y,
                                              const Triangle &tri) {
    double lambda1, lambda2, lambda3;
    computeBarycentricCoords(x, y, tri, lambda1, lambda2, lambda3);

    // Point is inside if all barycentric coordinates are between 0 and 1
    const double EPSILON = 1e-10; // To handle numerical precision
    return (lambda1 >= -EPSILON && lambda1 <= 1 + EPSILON &&
            lambda2 >= -EPSILON && lambda2 <= 1 + EPSILON &&
            lambda3 >= -EPSILON && lambda3 <= 1 + EPSILON);
}

void ImplicitFunctionBuilder::computeBarycentricCoords(double x, double y,
                                                       const Triangle &tri,
                                                       double &lambda1,
                                                       double &lambda2,
                                                       double &lambda3) {
    // Get vertex coordinates
    const Vertex &v1 = vertices[tri.v1];
    const Vertex &v2 = vertices[tri.v2];
    const Vertex &v3 = vertices[tri.v3];

    // Compute area of the full triangle using cross product
    double area =
        ((v2.x - v1.x) * (v3.y - v1.y) - (v3.x - v1.x) * (v2.y - v1.y)) / 2.0;

    // Handle degenerate triangle
    if (std::abs(area) < 1e-10) {
        throw std::runtime_error("Degenerate triangle detected");
    }

    // Compute barycentric coordinates using subtriangle areas
    // lambda1 is the ratio of the area of the subtriangle formed by point P
    // and vertices v2,v3 to the area of the full triangle
    lambda1 =
        ((v2.x - x) * (v3.y - y) - (v3.x - x) * (v2.y - y)) / (2.0 * area);

    // lambda2 for subtriangle of P, v3, v1
    lambda2 =
        ((v3.x - x) * (v1.y - y) - (v1.x - x) * (v3.y - y)) / (2.0 * area);

    // lambda3 for subtriangle of P, v1, v2
    lambda3 =
        ((v1.x - x) * (v2.y - y) - (v2.x - x) * (v1.y - y)) / (2.0 * area);
}

// Utility method to find which triangle contains a point
int ImplicitFunctionBuilder::findContainingTriangle(double x, double y) {
    for (size_t i = 0; i < triangles.size(); ++i) {
        if (pointInTriangle(x, y, triangles[i])) {
            return i;
        }
    }
    return -1; // Point not found in any triangle
}

// Utility method to create a DataPoint from arbitrary position
DataPoint
ImplicitFunctionBuilder::createDataPoint(double x, double y,
                                         const Eigen::Vector2d &normal) {
    DataPoint dp;
    dp.x = x;
    dp.y = y;
    dp.normal = normal;

    // Find containing triangle
    dp.containing_triangle = findContainingTriangle(x, y);
    if (dp.containing_triangle == -1) {
        throw std::runtime_error("Point not found in mesh");
    }

    // Compute barycentric coordinates
    const Triangle &tri = triangles[dp.containing_triangle];
    double lambda1, lambda2, lambda3;
    computeBarycentricCoords(x, y, tri, lambda1, lambda2, lambda3);
    dp.barycentric = Eigen::Vector3d(lambda1, lambda2, lambda3);

    return dp;
}
