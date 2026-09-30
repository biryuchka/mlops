#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

namespace {

void validate(const py::array& value, const char* name) {
    if (!value.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error(std::string(name) + ": expected native-endian float64 dtype, got " +
                             py::str(value.dtype()).cast<std::string>());
    }
    if (value.ndim() != 3) {
        throw py::value_error(std::string(name) + ": expected 3 dimensions, got " +
                              std::to_string(value.ndim()));
    }
    if (!(value.flags() & py::array::c_style)) {
        throw py::value_error(std::string(name) + ": array must be C-contiguous");
    }
    if (!value.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error(std::string(name) + ": array must be aligned");
    }
}

void check_same_shape(const py::array& reference, const py::array& value, const char* name) {
    for (py::ssize_t axis = 0; axis < 3; ++axis) {
        if (value.shape(axis) != reference.shape(axis)) {
            throw py::value_error(std::string(name) + ": shape must match a (broadcasting is not supported)");
        }
    }
}

}  // namespace

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    validate(a, "a");
    validate(b, "b");
    validate(c, "c");
    check_same_shape(a, b, "b");
    check_same_shape(a, c, "c");

    py::array_t<double> out({a.shape(0), a.shape(1), a.shape(2)});
    const auto size = static_cast<std::size_t>(a.size());
    if (size > 0) {
        mac_kernel(static_cast<const double*>(a.data()),
                   static_cast<const double*>(b.data()),
                   static_cast<const double*>(c.data()),
                   out.mutable_data(), size);
    }
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac,
               "Return a * b + c elementwise for three C-contiguous, aligned 3D float64 arrays",
               py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("c").noconvert());
}
