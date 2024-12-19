#include <algorithm>
#include <fstream>
#include <implicit2d/ImplicitFunctionBuilder.h>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>

/*
- Add visualization code to see the results?
- Modify the synthetic data pattern?
- Add different types of faults?
- Explain any specific part in more detail?
*/

ImplicitFunctionBuilder::ImplicitFunctionBuilder() {}

ImplicitFunctionBuilder::ImplicitFunctionBuilder(
    const std::vector<Vertex> &vertices, const std::vector<Triangle> &triangles,
    const std::vector<Edge> &edges, const std::vector<DataPoint> &data_points)
    : vertices(vertices), triangles(triangles), edges(edges),
      data_points(data_points) {
    // Precompute gradients of basis functions for each triangle
    precomputeTriangleBasis();
}

void ImplicitFunctionBuilder::beginDescription() {
    // Initialize the linear system
    A.setZero();
    b.setZero();
}

void ImplicitFunctionBuilder::addVertex(double x, double y) {
    vertices.push_back({x, y, (u_int32_t)vertices.size()});
}

void ImplicitFunctionBuilder::addVertices(const std::vector<double> &vertices) {
    if (vertices.size() % 2 != 0) {
        throw std::invalid_argument(
            "Vertices vector must have an even number of elements.");
    }
    this->vertices.reserve(vertices.size() / 2);
    uint32_t id = 0;
    for (size_t i = 0; i < vertices.size(); i += 2) {
        this->vertices.push_back({vertices[i], vertices[i + 1], id++});
    }
}

void ImplicitFunctionBuilder::addTriangle(uint32_t v1, uint32_t v2,
                                          uint32_t v3) {
    triangles.push_back({v1, v2, v3, (uint32_t)triangles.size()});
}

void ImplicitFunctionBuilder::addTriangles(
    const std::vector<uint32_t> &triangles) {
    if (triangles.size() % 3 != 0) {
        throw std::invalid_argument(
            "Triangles vector must have a multiple of 3 elements.");
    }
    this->triangles.reserve(triangles.size() / 3);
    uint32_t id = 0;
    for (size_t i = 0; i < triangles.size(); i += 3) {
        this->triangles.push_back(
            {triangles[i], triangles[i + 1], triangles[i + 2], id++});
    }
}

void ImplicitFunctionBuilder::addEdge(uint32_t v1, uint32_t v2) {
    edges.push_back({v1, v2, false, (uint32_t)edges.size()});
}

void ImplicitFunctionBuilder::addDataPoint(double x, double y,
                                           const Normal &value) {
    data_points.push_back({x, y, value});
}

void ImplicitFunctionBuilder::setDataPoints(
    const std::vector<DataPoint> &new_data_points) {
    data_points = new_data_points;
    // No need to call precomputeTriangleBasis() as it depends only on
    // triangles
}

void ImplicitFunctionBuilder::endDescription() {
    // Build the linear system that will solve for the implicit function
    precomputeTriangleBasis();
    buildLinearSystem();
    solve();
}

// Build the linear system that will solve for the implicit function
void ImplicitFunctionBuilder::buildLinearSystem() {
    int n = vertices.size();
    int m = data_points.size(); // number of data point constraints

    // Size the system: vertex equations + data point equations + 1 anchor
    // equation
    A.resize(n + m + 1, n);
    b.resize(n + m + 1);
    b.setZero();

    std::cout << "System size: " << (n + m + 1) << " equations, " << n
              << " unknowns\n";
    std::cout << "Matrix A size: " << A.rows() << "x" << A.cols() << "\n";
    std::cout << "Vector b size: " << b.size() << "\n";

    // Get edge connectivity respecting cuts (no edges across faults)
    auto edge_neighbors = buildEdgeConnectivity();

    // Build sparse matrix structure
    std::vector<Eigen::Triplet<double>> triplets;

    // First n equations: Laplacian smoothing for all vertices
    for (int i = 0; i < n; i++) {
        double sum_weights = 0.0;
        const auto &neighbors = edge_neighbors[i];

        // Add neighbor contributions with weight 1
        for (int j : neighbors) {
            triplets.emplace_back(i, j, 1.0);
            sum_weights += 1.0;
        }

        // Add diagonal term (negative sum of weights)
        if (sum_weights > 0) { // Only add if vertex has neighbors
            triplets.emplace_back(i, i, -sum_weights);
        }
        // b[i] = 0.0 (already set by setZero)
    }

    // Next m equations: gradient constraints from data points
    for (int k = 0; k < m; k++) {
        int eq = n + k; // current equation number
        const DataPoint &dp = data_points[k];
        const Triangle &tri = triangles[dp.containing_triangle];
        const auto &basis_grads =
            triangle_basis_gradients[dp.containing_triangle];

        std::cout << "Data point " << k << " in triangle "
                  << dp.containing_triangle << " with normal "
                  << dp.normal.transpose() << "\n";

        // Contribution from each vertex of the containing triangle
        double weight = 1.0; // Can be adjusted based on data confidence
        triplets.emplace_back(eq, tri.v1,
                              weight * dp.normal.dot(basis_grads.col(0)));
        triplets.emplace_back(eq, tri.v2,
                              weight * dp.normal.dot(basis_grads.col(1)));
        triplets.emplace_back(eq, tri.v3,
                              weight * dp.normal.dot(basis_grads.col(2)));

        // RHS: we want grad(f) · n = 1
        b[eq] = weight;
    }

    // Last equation: anchor one vertex (e.g., first vertex) to fix the constant
    triplets.emplace_back(n + m, 0, 1.0);
    // b[n + m] = 0.0 (already set by setZero)

    // Build sparse matrix from triplets
    A.setFromTriplets(triplets.begin(), triplets.end());

    // Print system statistics
    std::cout << "Number of non-zeros in A: " << triplets.size() << "\n";
    double density = double(triplets.size()) / (A.rows() * A.cols()) * 100;
    std::cout << "Matrix density: " << density << "%\n";
}

// Solve the linear system to get the implicit function values
const Eigen::VectorXd &ImplicitFunctionBuilder::solve() {
    std::cout << "Solving system of size " << A.rows() << "x" << A.cols()
              << "\n";

    // Use SparseLU for the normal equations: (A^T * A)x = A^T * b
    Eigen::SparseMatrix<double> AtA = A.transpose() * A;
    Eigen::VectorXd Atb = A.transpose() * b;

    std::cout << "Normal equations size: " << AtA.rows() << "x" << AtA.cols()
              << "\n";

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(AtA);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to decompose matrix");
    }

    solution_ = solver.solve(Atb);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Failed to solve system");
    }

    std::cout << "Solution norm: " << solution_.norm() << "\n";
    return solution_;
}

// Evaluate the implicit function at any point using linear interpolation
double ImplicitFunctionBuilder::evaluate(double x, double y) const {
    // Find containing triangle using point location
    for (const Triangle &tri : triangles) {
        if (pointInTriangle(x, y, tri)) {
            // Compute barycentric coordinates
            double lambda1, lambda2, lambda3;
            computeBarycentricCoords(x, y, tri, lambda1, lambda2, lambda3);

            // Interpolate function value
            return lambda1 * solution_[tri.v1] + lambda2 * solution_[tri.v2] +
                   lambda3 * solution_[tri.v3];
        }
    }
    throw std::runtime_error("Point not found in mesh");
}

// Helper function to build edge connectivity
std::vector<std::vector<uint32_t>>
ImplicitFunctionBuilder::buildEdgeConnectivity() {
    std::vector<std::vector<uint32_t>> edge_neighbors(vertices.size());

    if (edges.empty()) {
        // No edges provided - build connectivity from triangles
        for (const Triangle &tri : triangles) {
            // Add all triangle edges to connectivity
            edge_neighbors[tri.v1].push_back(tri.v2);
            edge_neighbors[tri.v1].push_back(tri.v3);
            edge_neighbors[tri.v2].push_back(tri.v1);
            edge_neighbors[tri.v2].push_back(tri.v3);
            edge_neighbors[tri.v3].push_back(tri.v1);
            edge_neighbors[tri.v3].push_back(tri.v2);
        }

        // Remove duplicates in each vertex's neighbor list
        for (auto &neighbors : edge_neighbors) {
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                            neighbors.end());
        }
    } else {
        // Use provided edges
        for (const Edge &edge : edges) {
            if (!edge.is_cut) {
                edge_neighbors[edge.v1].push_back(edge.v2);
                edge_neighbors[edge.v2].push_back(edge.v1);
            }
        }
    }

    return edge_neighbors;
}

bool ImplicitFunctionBuilder::pointInTriangle(double x, double y,
                                              const Triangle &tri) const {
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
                                                       double &lambda3) const {
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
int ImplicitFunctionBuilder::findContainingTriangle(double x, double y) const {
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

// Compute gradients of barycentric basis functions for each triangle
void ImplicitFunctionBuilder::precomputeTriangleBasis() {
    triangle_basis_gradients.resize(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) {
        const Triangle &tri = triangles[i];
        const Vertex &v1 = vertices[tri.v1];
        const Vertex &v2 = vertices[tri.v2];
        const Vertex &v3 = vertices[tri.v3];

        // Compute area and gradients of barycentric basis functions
        double area =
            ((v2.x - v1.x) * (v3.y - v1.y) - (v3.x - v1.x) * (v2.y - v1.y)) /
            2.0;

        // Store gradients of barycentric basis functions
        Eigen::Matrix<double, 2, 3> gradients;
        gradients.col(0) = Eigen::Vector2d((v2.y - v3.y) / (2 * area),
                                           (v3.x - v2.x) / (2 * area));
        gradients.col(1) = Eigen::Vector2d((v3.y - v1.y) / (2 * area),
                                           (v1.x - v3.x) / (2 * area));
        gradients.col(2) = Eigen::Vector2d((v1.y - v2.y) / (2 * area),
                                           (v2.x - v1.x) / (2 * area));

        triangle_basis_gradients[i] = gradients;
    }
}

std::vector<std::vector<std::pair<double, double>>>
ImplicitFunctionBuilder::computeContours(double min_val, double max_val,
                                         int num_contours) const {
    std::vector<std::vector<std::pair<double, double>>> contours;

    // Compute contour levels
    std::vector<double> levels;
    double step = (max_val - min_val) / (num_contours - 1);
    for (int i = 0; i < num_contours; ++i) {
        levels.push_back(min_val + i * step);
    }

    // For each triangle
    for (const auto &tri : triangles) {
        // Get vertex values
        double v1_val = solution_[tri.v1];
        double v2_val = solution_[tri.v2];
        double v3_val = solution_[tri.v3];

        // Get vertex coordinates
        const auto &p1 = vertices[tri.v1];
        const auto &p2 = vertices[tri.v2];
        const auto &p3 = vertices[tri.v3];

        // For each contour level
        for (double level : levels) {
            std::vector<std::pair<double, double>> intersections;

            // Check each edge for intersection
            auto checkEdge = [&](const Vertex &va, const Vertex &vb,
                                 double val_a, double val_b) {
                if ((val_a - level) * (val_b - level) < 0) {
                    double t = (level - val_a) / (val_b - val_a);
                    double x = va.x + t * (vb.x - va.x);
                    double y = va.y + t * (vb.y - va.y);
                    intersections.push_back({x, y});
                }
            };

            checkEdge(p1, p2, v1_val, v2_val);
            checkEdge(p2, p3, v2_val, v3_val);
            checkEdge(p3, p1, v3_val, v1_val);

            if (intersections.size() == 2) {
                contours.push_back(intersections);
            }
        }
    }

    return contours;
}

/**
 * This will create an SVG file showing:
 * - The mesh (gray lines)
 * - The fault(s) (red lines)
 * - Data points (blue dots)
 * - Normal vectors at data points (blue lines)
 * - Contours of the implicit function (gray lines)
 *
 * The visualization is scaled by 100 to make the SVG more viewable (since we're
 * working with coordinates around 1.0).
 */
void ImplicitFunctionBuilder::visualize(
    const std::string &filename, const VisualizationOptions &options) const {

    // Determine solution range
    double min_val = solution_.minCoeff();
    double max_val = solution_.maxCoeff();

    // Compute contours
    auto contours = computeContours(min_val, max_val, options.nbIsos);

    // Determine bounding box
    double min_x = vertices[0].x;
    double max_x = vertices[0].x;
    double min_y = vertices[0].y;
    double max_y = vertices[0].y;

    for (const auto &v : vertices) {
        min_x = std::min(min_x, v.x);
        max_x = std::max(max_x, v.x);
        min_y = std::min(min_y, v.y);
        max_y = std::max(max_y, v.y);
    }

    // Add some padding
    double padding = 0;
    double width = max_x - min_x;
    double height = max_y - min_y;
    min_x -= padding * width;
    max_x += padding * width;
    min_y -= padding * height;
    max_y += padding * height;

    // SVG setup with viewBox
    std::ofstream svg(filename);
    svg << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"" << min_x * 100
        << " " << min_y * 100 << " " << (max_x - min_x) * 100 << " "
        << (max_y - min_y) * 100 << "\">\n";

    // Style definitions
    svg << "<defs>\n"
        << "  <style>\n"
        << "    .mesh-edge { stroke: #ccc; stroke-width: 0.5; }\n"
        << "    .fault { stroke: red; stroke-width: 2; }\n"
        << "    .data-point { fill: blue; }\n"
        << "    .normal { stroke: blue; stroke-width: 1; }\n"
        << "    .contour { stroke: #666; stroke-width: 0.5; fill: none; }\n"
        << "    .bbox { stroke: #000; stroke-width: 1; fill: none; }\n" // Add
                                                                        // bbox
                                                                        // style
        << "  </style>\n"
        << "</defs>\n";

    // Draw bounding box (before mesh so it's behind everything)
    if (options.showBBox) {
        svg << "<rect class=\"bbox\" "
            << "x=\"" << min_x * 100 << "\" "
            << "y=\"" << min_y * 100 << "\" "
            << "width=\"" << (max_x - min_x) * 100 << "\" "
            << "height=\"" << (max_y - min_y) * 100 << "\"/>\n";
    }

    // Draw mesh edges
    if (options.showMesh) {
        svg << "<g class=\"mesh\">\n";
        for (const auto &edge : edges) {
            const auto &v1 = vertices[edge.v1];
            const auto &v2 = vertices[edge.v2];
            svg << "<line class=\"" << (edge.is_cut ? "fault" : "mesh-edge")
                << "\" x1=\"" << v1.x * 100 << "\" y1=\"" << v1.y * 100
                << "\" x2=\"" << v2.x * 100 << "\" y2=\"" << v2.y * 100
                << "\"/>\n";
        }
        svg << "</g>\n";
    }

    // Draw contours
    if (options.showContours) {
        svg << "<g class=\"contours\">\n";
        for (const auto &contour : contours) {
            if (contour.size() >= 2) {
                svg << "<line class=\"contour\" "
                    << "x1=\"" << contour[0].first * 100 << "\" "
                    << "y1=\"" << contour[0].second * 100 << "\" "
                    << "x2=\"" << contour[1].first * 100 << "\" "
                    << "y2=\"" << contour[1].second * 100 << "\"/>\n";
            }
        }
        svg << "</g>\n";
    }

    // Draw data points and normals
    if (options.showDataPoints) {
        svg << "<g class=\"data-points\">\n";
        double normal_length = 0.5; // Length of normal vectors
        for (const auto &dp : data_points) {
            svg << "<circle class=\"data-point\" cx=\"" << dp.x * 100
                << "\" cy=\"" << dp.y * 100 << "\" r=\"3\"/>\n";

            // Draw normal vector
            double nx = dp.normal[0] * normal_length;
            double ny = dp.normal[1] * normal_length;
            svg << "<line class=\"normal\" x1=\"" << dp.x * 100 << "\" y1=\""
                << dp.y * 100 << "\" x2=\"" << (dp.x + nx) * 100 << "\" y2=\""
                << (dp.y + ny) * 100 << "\"/>\n";
        }
        svg << "</g>\n";
    }

    svg << "</svg>\n";
    svg.close();
}

void ImplicitFunctionBuilder::visualizeColorMap(const std::string &filename,
                                                int nx, int ny,
                                                bool showMesh) const {
    // Determine bounding box
    double min_x = vertices[0].x;
    double max_x = vertices[0].x;
    double min_y = vertices[0].y;
    double max_y = vertices[0].y;

    for (const auto &v : vertices) {
        min_x = std::min(min_x, v.x);
        max_x = std::max(max_x, v.x);
        min_y = std::min(min_y, v.y);
        max_y = std::max(max_y, v.y);
    }

    // Sample domain regularly
    double dx = (max_x - min_x) / nx;
    double dy = (max_y - min_y) / ny;
    std::vector<std::vector<double>> values(ny, std::vector<double>(nx));
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();

    // Sample the function
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double x = min_x + i * dx;
            double y = min_y + j * dy;
            try {
                double val = evaluate(x, y);
                values[j][i] = val;
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            } catch (const std::runtime_error &e) {
                values[j][i] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }

    // Create SVG
    std::ofstream svg(filename);
    svg << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"" << min_x * 100
        << " " << min_y * 100 << " " << (max_x - min_x) * 100 << " "
        << (max_y - min_y) * 100 << "\">\n";

    // Define color interpolation
    svg << "<defs>\n"
        << "  <style>\n"
        << "    .mesh-edge { stroke: #333; stroke-width: 0.5; opacity: 0.5; }\n"
        << "    .fault { stroke: red; stroke-width: 2; }\n"
        << "  </style>\n"
        << "</defs>\n";

    // Draw colored rectangles
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (!std::isnan(values[j][i])) {
                double t = (values[j][i] - min_val) / (max_val - min_val);

                // Create a blue to red colormap
                int r = static_cast<int>(255 * t);
                int b = static_cast<int>(255 * (1 - t));

                svg << "<rect x=\"" << (min_x + i * dx) * 100 << "\" y=\""
                    << (min_y + j * dy) * 100 << "\" width=\"" << dx * 100
                    << "\" height=\"" << dy * 100 << "\" fill=\"rgb(" << r
                    << ",0," << b << ")\"/>\n";
            }
        }
    }

    // Optionally draw mesh on top
    if (showMesh) {
        svg << "<g class=\"mesh\">\n";
        for (const auto &edge : edges) {
            const auto &v1 = vertices[edge.v1];
            const auto &v2 = vertices[edge.v2];
            svg << "<line class=\"" << (edge.is_cut ? "fault" : "mesh-edge")
                << "\" x1=\"" << v1.x * 100 << "\" y1=\"" << v1.y * 100
                << "\" x2=\"" << v2.x * 100 << "\" y2=\"" << v2.y * 100
                << "\"/>\n";
        }
        svg << "</g>\n";
    }

    // Add colorbar
    double colorbar_width = (max_x - min_x) * 0.05;
    double colorbar_height = (max_y - min_y) * 0.8;
    double colorbar_x = max_x - colorbar_width * 2;
    double colorbar_y = min_y + (max_y - min_y) * 0.1;

    // Draw colorbar gradient
    int nsamples = 100;
    for (int i = 0; i < nsamples; ++i) {
        double t = static_cast<double>(i) / (nsamples - 1);
        int r = static_cast<int>(255 * t);
        int b = static_cast<int>(255 * (1 - t));

        svg << "<rect x=\"" << colorbar_x * 100 << "\" y=\""
            << (colorbar_y + t * colorbar_height) * 100 << "\" width=\""
            << colorbar_width * 100 << "\" height=\""
            << (colorbar_height / nsamples) * 100 << "\" fill=\"rgb(" << r
            << ",0," << b << ")\"/>\n";
    }

    // Add colorbar labels
    svg << "<text x=\"" << (colorbar_x + colorbar_width * 1.2) * 100
        << "\" y=\"" << colorbar_y * 100
        << "\" font-family=\"Arial\" font-size=\"12\">" << max_val
        << "</text>\n";

    svg << "<text x=\"" << (colorbar_x + colorbar_width * 1.2) * 100
        << "\" y=\"" << (colorbar_y + colorbar_height) * 100
        << "\" font-family=\"Arial\" font-size=\"12\">" << min_val
        << "</text>\n";

    svg << "</svg>\n";
    svg.close();
}