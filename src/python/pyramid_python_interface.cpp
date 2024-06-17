#include "../ome_tiff_to_chunked_pyramid.h"
#include "../pyramid_view.h"
#include "../utilities.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(libargolid, m) {
    py::class_<argolid::OmeTiffToChunkedPyramid, std::shared_ptr<argolid::OmeTiffToChunkedPyramid>>(m, "OmeTiffToChunkedPyramidCPP") \
    .def(py::init<>()) \
    .def("GenerateFromSingleFile", &argolid::OmeTiffToChunkedPyramid::GenerateFromSingleFile) \
    .def("GenerateFromCollection", &argolid::OmeTiffToChunkedPyramid::GenerateFromCollection) \
    .def("SetLogLevel", &argolid::OmeTiffToChunkedPyramid::SetLogLevel) ;

    py::class_<argolid::PyramidView, std::shared_ptr<argolid::PyramidView>>(m, "PyramidViewCPP") \
    .def(py::init<std::string_view, std::string, std::string, std::string, argolid::image_map&>()) \
    .def("GeneratePyramid", &argolid::PyramidView::GeneratePyramid);

    py::enum_<argolid::VisType>(m, "VisType")
        .value("NG_Zarr", argolid::VisType::NG_Zarr)
        .value("PCNG", argolid::VisType::PCNG)
        .value("Viv", argolid::VisType::Viv)
        .export_values();

    py::enum_<argolid::DSType>(m, "DSType")
        .value("Mode_Max", argolid::DSType::Mode_Max)
        .value("Mode_Min", argolid::DSType::Mode_Min)
        .value("Mean", argolid::DSType::Mean)
        .export_values();
}