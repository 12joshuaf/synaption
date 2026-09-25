//#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>
//#include "../core/tensor.h"
//#include "../core/network.h"

//namespace py = pybind11;
//using namespace synaption;
//
//PYBIND11_MODULE(synaption_core, m) {
//    m.doc() = "synaption native core (C++ neural net engine)";
//
//    py::class_<Tensor>(m, "Tensor")
//        .def(py::init<std::vector<size_t>>())
//        .def("shape", &Tensor::shape)
//        .def("size", &Tensor::size)
//        .def("__getitem__", [](Tensor& t, size_t i) { return t.at(i); })
//        .def("__setitem__", [](Tensor& t, size_t i, float v) { t.at(i) = v; });
//
//    py::class_<Network>(m, "Network")
//        .def(py::init<>())
//        .def("add_layer", &Network::add_layer)
//        .def("forward", &Network::forward);
//}