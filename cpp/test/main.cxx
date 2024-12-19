#include <cmath>
#include <implicit2d/ImplicitFunctionBuilder.h>
#include <iostream>
#include <vector>

using Mesh = std::pair<Vertices, Triangles>;
using Builder = ImplicitFunctionBuilder;

// Helper function to create a regular grid of triangles
Mesh createGrid(int nx, int ny, double width = 10.0, double height = 10.0) {

    Vertices vertices;
    Triangles triangles;

    // Create vertices
    double dx = width / (nx - 1);
    double dy = height / (ny - 1);
    uint32_t vid = 0;

    for (uint32_t j = 0; j < ny; j++) {
        for (uint32_t i = 0; i < nx; i++) {
            vertices.push_back({i * dx, j * dy, vid++});
        }
    }

    // Create triangles
    uint32_t tid = 0;
    for (uint32_t j = 0; j < ny - 1; j++) {
        for (uint32_t i = 0; i < nx - 1; i++) {
            uint32_t v1 = j * nx + i;
            uint32_t v2 = v1 + 1;
            uint32_t v3 = v1 + nx;
            uint32_t v4 = v2 + nx;

            // Lower triangle
            triangles.push_back({v1, v2, v3, tid++});
            // Upper triangle
            triangles.push_back({v2, v4, v3, tid++});
        }
    }

    return {vertices, triangles};
}

// Create fault edges (vertical fault in this example)
Edges createFaults(const Vertices &vertices, uint32_t nx, uint32_t ny,
                   double fault_x) {

    Edges edges;
    uint32_t eid = 0;

    // Create all edges
    for (uint32_t j = 0; j < ny - 1; j++) {
        for (uint32_t i = 0; i < nx - 1; i++) {
            uint32_t v1 = j * nx + i;
            uint32_t v2 = v1 + 1;
            uint32_t v3 = v1 + nx;

            // Check if edge crosses fault line
            bool is_fault =
                (vertices[v1].x <= fault_x && vertices[v2].x > fault_x);

            // Horizontal edges
            edges.push_back({v1, v2, is_fault, eid++});

            // Vertical edges
            edges.push_back({v1, v3, false, eid++});

            // Diagonal edges (for better mesh quality)
            if (i < nx - 1 && j < ny - 1) {
                edges.push_back({v1, v2 + nx, false, eid++});
            }
        }
    }

    // Add last row of horizontal edges
    for (uint32_t i = 0; i < nx - 1; i++) {
        uint32_t v1 = (ny - 1) * nx + i;
        edges.push_back({v1, v1 + 1, false, eid++});
    }

    // Add last column of vertical edges
    for (uint32_t j = 0; j < ny - 1; j++) {
        uint32_t v1 = j * nx + (nx - 1);
        edges.push_back({v1, v1 + nx, false, eid++});
    }

    return edges;
}

// Create synthetic data points with geological layer orientations
DataPoints createSyntheticData(Builder &builder, const Vertices &vertices,
                               double fault_x) {

    DataPoints data_points;

    // Create a folded structure with displacement across fault
    auto addDataPoint = [&](double x, double y, double angle) {
        Eigen::Vector2d normal(std::sin(angle), std::cos(angle));
        try {
            DataPoint dp = builder.createDataPoint(x, y, normal);
            data_points.push_back(dp);
        } catch (const std::runtime_error &e) {
            std::cerr << "Failed to add data point at (" << x << "," << y << ")"
                      << std::endl;
        }
    };

    // Left side of fault
    addDataPoint(2.0, 2.0, M_PI / 6); // 30 degrees
    addDataPoint(3.0, 5.0, M_PI / 4); // 45 degrees
    addDataPoint(4.0, 8.0, M_PI / 3); // 60 degrees

    // Right side of fault (with displacement)
    addDataPoint(7.0, 1.0, M_PI / 4); // 45 degrees
    addDataPoint(8.0, 4.0, M_PI / 3); // 60 degrees
    addDataPoint(9.0, 7.0, M_PI / 2); // 90 degrees

    return data_points;
}

int main() {
    // Create a 20x20 grid
    int nx = 20, ny = 20;
    auto [vertices, triangles] = createGrid(nx, ny);

    // Create fault at x = 5.0
    Edges edges = createFaults(vertices, nx, ny, 5.0);

    // Initialize builder
    Builder builder(vertices, triangles, edges, {});

    // Create synthetic data
    auto data_points = createSyntheticData(builder, vertices, 5.0);
    builder.setDataPoints(data_points);

    // Build and solve system
    builder.buildLinearSystem();
    Eigen::VectorXd solution = builder.solve();

    // Output some results
    std::cout << "Solution computed successfully\n";
    std::cout << "Number of vertices: " << vertices.size() << "\n";
    std::cout << "Number of triangles: " << triangles.size() << "\n";
    std::cout << "Number of edges: " << edges.size() << "\n";
    std::cout << "Number of data points: " << data_points.size() << "\n";

    builder.visualize("result.svg", VisualizationOptions{.showContours = true});
    builder.visualizeColorMap("result_colormap_hires.svg", 200, 200, false);

    // Sample some points across the domain
    if (0) {
        std::cout << "\nSampling implicit function values:\n";
        uint N = 100;
        for (double x = 0; x <= N; x += 2.5) {
            for (double y = 0; y <= N; y += 2.5) {
                try {
                    double val = builder.evaluate(x / 10, y / 10);
                    std::cout << "f(" << x << "," << y << ") = " << val << "\n";
                } catch (const std::runtime_error &e) {
                    std::cout << "Point (" << x << "," << y
                              << ") outside mesh\n";
                }
            }
        }
    }

    return 0;
}