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

#include "pyramid_view.h"
#include "chunked_base_to_pyr_gen.h"
#include "../utilities/utilities.h"
#include "pugixml.hpp"
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
namespace fs = std::filesystem;

using::tensorstore::Context;
using::tensorstore::internal_zarr::ChooseBaseDType;

namespace argolid {

  void PyramidView::AssembleBaseLevel(VisType v, const image_map& coordinate_map, const std::string& zarr_array_path) {
     if (v!=VisType::NG_Zarr && v!=VisType::Viv) {
      PLOG_INFO << "Unsupported Pyramid type requested";
      return;
    }

    int grid_x_max = 0, grid_y_max = 0, grid_c_max = 0;

    int img_count = 0;
    for (const auto & [name, location]: coordinate_map) {
      const auto[gx, gy, gc] = location;
      gc > grid_c_max ? grid_c_max = gc : grid_c_max = grid_c_max;
      gx > grid_x_max ? grid_x_max = gx : grid_x_max = grid_x_max;
      gy > grid_y_max ? grid_y_max = gy : grid_y_max = grid_y_max;
      ++img_count;
    }
    PLOG_DEBUG << "Total images found: " << img_count << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    auto [x_dim, y_dim, c_dim, num_dims] = GetZarrParams(v);

    ImageInfo whole_image;

    if (img_count != 0) {
      size_t write_failed_count = 0;
      const auto & sample_tiff_file = image_coll_path + "/" + coordinate_map.begin() -> first;
      TENSORSTORE_CHECK_OK_AND_ASSIGN(auto test_source, tensorstore::Open(
        GetOmeTiffSpecToRead(sample_tiff_file),
        tensorstore::OpenMode::open,
        tensorstore::ReadWriteMode::read).result());
      auto test_image_shape = test_source.domain().shape();

      whole_image._chunk_size_x = test_image_shape[4] + 2*x_spacing;
      whole_image._chunk_size_y = test_image_shape[3] + 2*y_spacing;
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
      new_image_shape[c_dim] = whole_image._num_channels;

      auto output_spec = [&test_source, &new_image_shape, &chunk_shape, &zarr_array_path, this]() {
          return GetZarrSpecToWrite(zarr_array_path, new_image_shape, chunk_shape, ChooseBaseDType(test_source.dtype()).value().encoded_dtype);
      }();

      TENSORSTORE_CHECK_OK_AND_ASSIGN(auto dest, tensorstore::Open(
        output_spec,
        tensorstore::OpenMode::create |
        tensorstore::OpenMode::delete_existing,
        tensorstore::ReadWriteMode::write).result());

      auto t4 = std::chrono::high_resolution_clock::now();
      for (const auto & [file_name, location]: coordinate_map) {
        th_pool.detach_task([ &dest, file_name=file_name, location=location, x_dim=x_dim, y_dim=y_dim, c_dim=c_dim, v, &whole_image, this]() {

          TENSORSTORE_CHECK_OK_AND_ASSIGN(auto source, tensorstore::Open(
            GetOmeTiffSpecToRead(image_coll_path + "/" + file_name),
            tensorstore::OpenMode::open,
            tensorstore::ReadWriteMode::read).result());
          PLOG_DEBUG << "Opening " << file_name;
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
          if (v == VisType::NG_Zarr) {
            transform = (std::move(transform) | tensorstore::Dims(c_dim).SizedInterval(c_grid, 1) |
              tensorstore::Dims(y_dim).SizedInterval(y_grid * whole_image._chunk_size_y + y_spacing, image_height) |
              tensorstore::Dims(x_dim).SizedInterval(x_grid * whole_image._chunk_size_x + x_spacing, image_width)).value();
          } else if (v == VisType::Viv) {
            transform = (std::move(transform) | tensorstore::Dims(c_dim).SizedInterval(c_grid, 1) |
              tensorstore::Dims(y_dim).SizedInterval(y_grid * whole_image._chunk_size_y + y_spacing, image_height) |
              tensorstore::Dims(x_dim).SizedInterval(x_grid * whole_image._chunk_size_x + x_spacing, image_width)).value();
          }
          tensorstore::Write(array, dest | transform).value();
        });
      }

      th_pool.wait();
    }
    base_image = whole_image;
  }

  void PyramidView::GeneratePyramid(const image_map& map, 
                                    VisType v, 
                                    int min_dim,  
                                    const std::unordered_map<std::int64_t, DSType>& channel_ds_config)
  {
    const auto image_dir = pyramid_zarr_path + "/" + image_name +".zarr";
    if (fs::exists(image_dir)) fs::remove_all(image_dir);
    PLOG_INFO << "GeneratePyramid Start ";
    if (v!=VisType::NG_Zarr && v!=VisType::Viv) {
      PLOG_INFO << "Unsupported Pyramid type requested";
      return;
    }


    const auto output_zarr_path = [v, &image_dir, this](){
      if (v==VisType::Viv){
        return image_dir +"/data.zarr/0";
      } else {
        return image_dir +"/0";
      }
    }();
    PLOG_INFO << "Starting to generate base layer ";
    AssembleBaseLevel(v, map, output_zarr_path+"/0") ;
    PLOG_INFO << "Finished generating base layer ";

    // generate pyramid
    ChunkedBaseToPyramid base_to_pyramid;
    int base_level_key = 0;
    int max_level = static_cast<int>(ceil(log2(std::max({base_image._full_image_width, base_image._full_image_width}))));
    int min_level = static_cast<int>(ceil(log2(min_dim)));
    auto max_level_key = max_level-min_level+1;
    PLOG_INFO << "Starting to generate pyramid ";
    base_to_pyramid.CreatePyramidImages(output_zarr_path, output_zarr_path, base_level_key, min_dim, v, channel_ds_config, th_pool);
    PLOG_INFO << "Finished generating pyramid ";

    // generate metadata
    WriteMultiscaleMetadataForImageCollection(image_name, pyramid_zarr_path, base_level_key, max_level_key, v, base_image);
    PLOG_INFO << "GeneratePyramid end ";
  }

} // ns argolid