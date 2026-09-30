#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../core/tensor.h"
#include "../core/activation.h"
#include "../core/loss.h"
#include "../core/optimizer.h"
#include "../core/network.h"

namespace py = pybind11;
using namespace synaption;

PYBIND11_MODULE(synaption_core, m) {
    m.doc() = "synaption native core (C++ neural net engine)";

    py::enum_<Activation>(m, "Activation")
        .value("Identity", Activation::Identity)
        .value("ReLU", Activation::ReLU)
        .value("Sigmoid", Activation::Sigmoid)
        .value("Tanh", Activation::Tanh)
        .value("Swish", Activation::Swish)
        .value("GELU", Activation::GELU)
        .value("ELU", Activation::ELU);

    py::enum_<Loss>(m, "Loss")
        .value("MSE", Loss::MSE)
        .value("BCE", Loss::BCE)
        .value("CCE", Loss::CCE)
        .value("Hinge", Loss::Hinge);

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<size_t>>())
        .def("shape", &Tensor::shape)
        .def("size", &Tensor::size)
        .def("__getitem__", [](Tensor& t, size_t i) { return t.at(i); })
        .def("__setitem__", [](Tensor& t, size_t i, float v) { t.at(i) = v; });

    py::class_<SGD>(m, "SGD")
        .def(py::init<float>(), py::arg("learning_rate"));

    py::class_<Network>(m, "Network")
        .def(py::init<>())
        .def("add_layer", &Network::add_layer,
            py::arg("in_features"), py::arg("out_features"),
            py::arg("activation") = Activation::ReLU)
        .def("forward", &Network::forward)
        .def("train_epoch", &Network::train_epoch,
            py::arg("inputs"), py::arg("targets"), py::arg("loss_type"), py::arg("optimizer"));
}