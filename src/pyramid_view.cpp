#include <fstream>
#include <iostream>
#include <regex>
#include <vector>
#include <list>
#include <cstring>
#include <unordered_set>
#include <string>
#include <stdint.h>
#include <fstream>
#include <iostream>
#include <string>
#include <future>
#include <cmath>
#include <thread>
#include <cstdlib>
#include <optional>
#include <filesystem>

#include "tensorstore/tensorstore.h"
#include "tensorstore/context.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/zarr/dtype.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/open.h"
#include "filepattern/filepattern.h"
#include <nlohmann/json.hpp>

#include "pyramid_view.h"
#include "chunked_base_to_pyr_gen.h"
#include "utilities.h"
#include "pugixml.hpp"
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;
namespace fs = std::filesystem;

using::tensorstore::Context;
using::tensorstore::internal_zarr::ChooseBaseDType;

namespace argolid {

  void PyramidView::AssembleBaseLevel(VisType v) {

    int grid_x_max = 0, grid_y_max = 0, grid_c_max = 0;

    int img_count = 0;
    for (const auto & [name, location]: base_image_map) {
      const auto[gx, gy, gc] = location;
      gc > grid_c_max ? grid_c_max = gc : grid_c_max = grid_c_max;
      gx > grid_x_max ? grid_x_max = gx : grid_x_max = grid_x_max;
      gy > grid_y_max ? grid_y_max = gy : grid_y_max = grid_y_max;
      ++img_count;
    }
    PLOG_INFO << "Total images found: " << img_count << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    auto [x_dim, y_dim, c_dim, num_dims] = GetZarrParams(v);

    ImageInfo whole_image;

    if (img_count != 0) {
      //std::list<tensorstore::WriteFutures> pending_writes;
      size_t write_failed_count = 0;
      const auto & sample_tiff_file = image_coll_path + "/" + base_image_map.begin() -> first;
      TENSORSTORE_CHECK_OK_AND_ASSIGN(auto test_source, tensorstore::Open(
        GetOmeTiffSpecToRead(sample_tiff_file),
        tensorstore::OpenMode::open,
        tensorstore::ReadWriteMode::read).result());
      auto test_image_shape = test_source.domain().shape();

      whole_image._chunk_size_x = test_image_shape[4];
      whole_image._chunk_size_y = test_image_shape[3];
      whole_image._full_image_width = (grid_x_max + 1) * whole_image._chunk_size_x;
      whole_image._full_image_height = (grid_y_max + 1) * whole_image._chunk_size_y;
      whole_image._num_channels = grid_c_max + 1;

      std::vector < std::int64_t > new_image_shape(num_dims, 1);
      std::vector < std::int64_t > chunk_shape(num_dims, 1);
      new_image_shape[y_dim] = whole_image._full_image_height;
      new_image_shape[x_dim] = whole_image._full_image_width;
      chunk_shape[y_dim] = whole_image._chunk_size_y;
      chunk_shape[x_dim] = whole_image._chunk_size_x;
      whole_image._data_type = test_source.dtype().name();
      if (v == VisType::NG_Zarr || v == VisType::Viv) {
        new_image_shape[c_dim] = whole_image._num_channels;
      }

      auto output_spec = [ & ]() {
        if (v == VisType::NG_Zarr || v == VisType::Viv) {
          return GetZarrSpecToWrite(base_zarr_path, new_image_shape, chunk_shape, ChooseBaseDType(test_source.dtype()).value().encoded_dtype);
        } else {
          return tensorstore::Spec();
        }
      }();

      TENSORSTORE_CHECK_OK_AND_ASSIGN(auto dest, tensorstore::Open(
        output_spec,
        tensorstore::OpenMode::create |
        tensorstore::OpenMode::delete_existing,
        tensorstore::ReadWriteMode::write).result());

      auto t4 = std::chrono::high_resolution_clock::now();
      for (const auto & [file_name, location]: base_image_map) {
        th_pool.push_task([ & dest, file_name, location, x_dim, y_dim, c_dim, v, & whole_image, this]() {

          TENSORSTORE_CHECK_OK_AND_ASSIGN(auto source, tensorstore::Open(
            GetOmeTiffSpecToRead(image_coll_path + "/" + file_name),
            tensorstore::OpenMode::open,
            tensorstore::ReadWriteMode::read).result());
          PLOG_INFO << "Opening " << file_name;
          auto image_shape = source.domain().shape();
          auto image_width = image_shape[4];
          auto image_height = image_shape[3];
          auto array = tensorstore::AllocateArray({
              image_height,
              image_width
            }, tensorstore::c_order,
            tensorstore::value_init, source.dtype());

          // initiate a read
          tensorstore::Read(source |
            tensorstore::Dims(3).ClosedInterval(0, image_height - 1) |
            tensorstore::Dims(4).ClosedInterval(0, image_width - 1),
            array).value();

          const auto & [x_grid, y_grid, c_grid] = location;

          tensorstore::IndexTransform < > transform = tensorstore::IdentityTransform(dest.domain());
          if (v == VisType::PCNG) {
            transform = (std::move(transform) | tensorstore::Dims("z", "channel").IndexSlice({
                0,
                c_grid
              }) |
              tensorstore::Dims(y_dim).SizedInterval(y_grid * whole_image._chunk_size_y, image_height) |
              tensorstore::Dims(x_dim).SizedInterval(x_grid * whole_image._chunk_size_x, image_width) |
              tensorstore::Dims(x_dim, y_dim).Transpose({
                y_dim,
                x_dim
              })).value();

          } else if (v == VisType::NG_Zarr) {
            transform = (std::move(transform) | tensorstore::Dims(c_dim).SizedInterval(c_grid, 1) |
              tensorstore::Dims(y_dim).SizedInterval(y_grid * whole_image._chunk_size_y, image_height) |
              tensorstore::Dims(x_dim).SizedInterval(x_grid * whole_image._chunk_size_x, image_width)).value();
          } else if (v == VisType::Viv) {
            transform = (std::move(transform) | tensorstore::Dims(c_dim).SizedInterval(c_grid, 1) |
              tensorstore::Dims(y_dim).SizedInterval(y_grid * whole_image._chunk_size_y, image_height) |
              tensorstore::Dims(x_dim).SizedInterval(x_grid * whole_image._chunk_size_x, image_width)).value();
          }
          tensorstore::Write(array, dest | transform).value();
        });
      }

      th_pool.wait_for_tasks();
    }
    base_image = whole_image;
  }

  void PyramidView::AssembleBaseLevel(VisType v, image_map m, const std::string& output_path) {

    auto [x_dim, y_dim, c_dim, num_dims] = GetZarrParams(v);

    auto input_spec = [v, this]() {
      if (v == VisType::NG_Zarr | v == VisType::Viv) {
        return GetZarrSpecToRead(base_zarr_path);
      } else { // this will probably never happen
        return tensorstore::Spec();
      }
    }();

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto base_store, tensorstore::Open(
      input_spec,
      tensorstore::OpenMode::open,
      tensorstore::ReadWriteMode::read).result());
    auto base_image_shape = base_store.domain().shape();
    auto read_chunk_shape = base_store.chunk_layout().value().read_chunk_shape();

    std::vector < std::int64_t > new_image_shape(num_dims, 1);
    std::vector < std::int64_t > chunk_shape(num_dims, 1);

    new_image_shape[y_dim] = base_image_shape[y_dim];
    new_image_shape[x_dim] = base_image_shape[x_dim];

    chunk_shape[y_dim] = read_chunk_shape[y_dim];
    chunk_shape[x_dim] = read_chunk_shape[x_dim];

    auto open_mode = tensorstore::OpenMode::create;
    if (v == VisType::NG_Zarr | v == VisType::Viv) {
      open_mode = open_mode | tensorstore::OpenMode::delete_existing;
      new_image_shape[c_dim] = base_image_shape[c_dim];
    }


    auto output_spec = [v, &output_path, &new_image_shape, & chunk_shape, & base_store, this]() {
      if (v == VisType::NG_Zarr) {
        return GetZarrSpecToWrite(output_path + "/0", new_image_shape, chunk_shape, ChooseBaseDType(base_store.dtype()).value().encoded_dtype);
      } else if (v == VisType::Viv) {
        return GetZarrSpecToWrite(output_path + "/0", new_image_shape, chunk_shape, ChooseBaseDType(base_store.dtype()).value().encoded_dtype);
      } else {
        return tensorstore::Spec();
      }
    }();

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto dest, tensorstore::Open(
      output_spec,
      open_mode,
      tensorstore::ReadWriteMode::write).result());

    size_t write_failed_count = 0;

    for (const auto & [file_name, location]: m) {
      // find where to read data from
      const auto base_location = [ & file_name, this]() -> std::optional < std::tuple < std::uint32_t,
        uint32_t, uint32_t >> {
          if (auto search = base_image_map.find(file_name); search != base_image_map.end()) {
            return std::optional {
              search -> second
            };
          } else {
            return std::nullopt;
          }
        }();

      if (!base_location.has_value()) {
        continue;
      }

      th_pool.push_task([ & base_store, & dest, file_name, location, base_location, x_dim, y_dim, c_dim, v, this]() {

        const auto & [x_grid_base, y_grid_base, c_grid_base] = base_location.value();

        tensorstore::IndexTransform < > read_transform = tensorstore::IdentityTransform(base_store.domain());

        if (v == VisType::NG_Zarr) {
          read_transform = (std::move(read_transform) | tensorstore::Dims(c_dim).SizedInterval(c_grid_base, 1) |
            tensorstore::Dims(y_dim).SizedInterval(y_grid_base * base_image._chunk_size_y, base_image._chunk_size_y) |
            tensorstore::Dims(x_dim).SizedInterval(x_grid_base * base_image._chunk_size_x, base_image._chunk_size_x)).value();
        } else if (v == VisType::Viv) {
          read_transform = (std::move(read_transform) | tensorstore::Dims(c_dim).SizedInterval(c_grid_base, 1) |
            tensorstore::Dims(y_dim).SizedInterval(y_grid_base * base_image._chunk_size_y, base_image._chunk_size_y) |
            tensorstore::Dims(x_dim).SizedInterval(x_grid_base * base_image._chunk_size_x, base_image._chunk_size_x)).value();
        }

        auto array = tensorstore::AllocateArray({
            base_image._chunk_size_y,
            base_image._chunk_size_x
          }, tensorstore::c_order,
          tensorstore::value_init, base_store.dtype());

        // initiate a read
        tensorstore::Read(base_store | read_transform, array).value();

        const auto & [x_grid, y_grid, c_grid] = location;

        tensorstore::IndexTransform < > write_transform = tensorstore::IdentityTransform(dest.domain());
        if (v == VisType::NG_Zarr) {
          write_transform = (std::move(write_transform) | tensorstore::Dims(c_dim).SizedInterval(c_grid, 1) |
            tensorstore::Dims(y_dim).SizedInterval(y_grid * base_image._chunk_size_y, base_image._chunk_size_y) |
            tensorstore::Dims(x_dim).SizedInterval(x_grid * base_image._chunk_size_x, base_image._chunk_size_x)).value();
        } else if (v == VisType::Viv) {
          write_transform = (std::move(write_transform) | tensorstore::Dims(c_dim).SizedInterval(c_grid, 1) |
            tensorstore::Dims(y_dim).SizedInterval(y_grid * base_image._chunk_size_y, base_image._chunk_size_y) |
            tensorstore::Dims(x_dim).SizedInterval(x_grid * base_image._chunk_size_x, base_image._chunk_size_x)).value();
        }
        tensorstore::Write(array, dest | write_transform).value();
      });
    }

    th_pool.wait_for_tasks();
  }

  void PyramidView::GeneratePyramid(std::optional<image_map> map, 
                                    VisType v, 
                                    int min_dim,  
                                    std::unordered_map<std::int64_t, DSType>& channel_ds_config)
  {
    const auto output_zarr_path = [v, this](){
      if (v==VisType::Viv){
        return pyramid_zarr_path + "/" + image_name +".zarr/data.zarr/0";
      } else {
        return pyramid_zarr_path + "/" + image_name +".zarr/0";
      }
    }();

    if (map.has_value()){
      AssembleBaseLevel(v,map.value(),output_zarr_path);
    } else {
      // copy base level zarr file
        fs::path destination{output_zarr_path+"/0"};
        if (!fs::exists(destination)) {
            fs::create_directories(destination);
        }

        // Iterate over files in the source directory
        fs::path source{base_zarr_path};
        for (const auto& entry : fs::directory_iterator(source)) {
            const auto& path = entry.path();
            auto destPath = destination / path.filename();

            // Copy file
            if (fs::is_regular_file(path)) {
                fs::copy_file(path, destPath, fs::copy_options::overwrite_existing);
            }
        }
    }

    // generate pyramid
    ChunkedBaseToPyramid base_to_pyramid;
    int base_level_key = 0;
    int max_level = static_cast<int>(ceil(log2(std::max({base_image._full_image_width, base_image._full_image_width}))));
    int min_level = static_cast<int>(ceil(log2(min_dim)));
    auto max_level_key = max_level-min_level+1;
    base_to_pyramid.CreatePyramidImages(output_zarr_path, output_zarr_path, base_level_key, min_dim, v, channel_ds_config, th_pool);
    
    // generate metadata
    WriteMultiscaleMetadataForImageCollection(image_name, pyramid_zarr_path, base_level_key, max_level_key, v, base_image);
  }

} // ns argolid