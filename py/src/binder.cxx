#include <CppLib/point.h>
#include <CppLib/triangle.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(pyalgo, m) {
    pybind11::class_<PointSource>(m, "PointSource")
        .def(pybind11::init<>())                             // ctor
        .def(pybind11::init<const Vec3D &, const Vec3D &>()) // ctor
        .def(pybind11::init<const Vec3D &, const Vec3D &, double,
                            double>()) // ctor
        .def("setPoisson", &PointSource::setPoisson)
        .def("setShear", &PointSource::setShear)
        .def("stress", &PointSource::stress)
        .def("stresses", &PointSource::stresses);

    pybind11::class_<TriangleSource>(m, "TriangleSource")
        .def(pybind11::init<const Point3D &, const Point3D &, const Point3D &,
                            const Vec3D &>()) // ctor
        .def(pybind11::init<const Point3D &, const Point3D &, const Point3D &,
                            const Vec3D &, double, double, uint8_t>()) // ctor
        .def("setPoisson", &TriangleSource::setPoisson)
        .def("setShear", &TriangleSource::setShear)
        .def("stress", &TriangleSource::stress)
        .def("stresses", &TriangleSource::stresses);
}
