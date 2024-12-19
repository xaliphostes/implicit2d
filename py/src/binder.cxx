#include <implicit2d/ImplicitFunctionBuilder.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

PYBIND11_MODULE(implicit2d, m) {
    pybind11::class_<ImplicitFunctionBuilder>(m, "ImplicitBuilder")
        .def(pybind11::init<>())
        .def(
            pybind11::init<const std::vector<Vertex> &,
                           const std::vector<Triangle>, const std::vector<Edge>,
                           const std::vector<DataPoint> &>())

        .def("beginDescription", &ImplicitFunctionBuilder::beginDescription)

        .def("addVertex", &ImplicitFunctionBuilder::addVertex)
        .def("addVertices", &ImplicitFunctionBuilder::addVertices)
        
        .def("addTriangle", &ImplicitFunctionBuilder::addTriangle)
        .def("addTriangles", &ImplicitFunctionBuilder::addTriangles)

        .def("addEdge", &ImplicitFunctionBuilder::addEdge)
        .def("addDataPoint", &ImplicitFunctionBuilder::addDataPoint)

        .def("endDescription", &ImplicitFunctionBuilder::endDescription);
}
