#include "chunked_pyramid_assembler.h"
#include "../utilities/utilities.h"
#include "pugixml.hpp"
#include "tiffio.h"
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

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
#include <climits>

#include "tensorstore/tensorstore.h"
#include "tensorstore/context.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/zarr/dtype.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/open.h"

#include "filepattern/filepattern.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <filesystem>
namespace fs = std::filesystem;

using ::tensorstore::Context;
using ::tensorstore::internal_zarr::ChooseBaseDType;

namespace argolid{
std::optional<std::int64_t> retrieve_val(const std::string& var, const Map& m){
    auto it = m.find(var);
    if(it != m.end()){
        if (std::holds_alternative<int>(it->second)){
          return static_cast<std::int64_t>(std::get<int>(it->second));
        } else {
          return {};
        }   
    } else {
        return {};
    }
}
ImageInfo OmeTiffCollToChunked::Assemble(const std::string& input_dir,
                                    const std::string& pattern ,
                                    const std::string& output_file, 
                                    const std::string& scale_key, 
                                    VisType v, 
                                    BS::thread_pool<BS::tp::none>& th_pool)
{
  int grid_x_max = 0, grid_y_max = 0, grid_c_max = 0;
  int grid_x_min = INT_MAX, grid_y_min = INT_MAX, grid_c_min = INT_MAX;
  std::vector<ImageSegment> image_vec;
  ImageInfo whole_image;

  auto fp = std::make_unique<FilePattern> (input_dir, pattern);
  auto files = fp->getFiles();

  for(const auto& [map, values]: files){
    auto gx = retrieve_val("x", map);
    auto gy = retrieve_val("y", map);
    if(gx.has_value() && gy.has_value()){
      auto gc = retrieve_val("c", map);
      if(!gc.has_value()) gc.emplace(0); // use default channel 0
      auto fname = values[0].string();  
      gc.value() > grid_c_max ? grid_c_max = gc.value() : grid_c_max = grid_c_max;
      gx.value() > grid_x_max ? grid_x_max = gx.value() : grid_x_max = grid_x_max;
      gy.value() > grid_y_max ? grid_y_max = gy.value() : grid_y_max = grid_y_max;

      gc.value() < grid_c_min ? grid_c_min = gc.value() : grid_c_min = grid_c_min;
      gx.value() < grid_x_min ? grid_x_min = gx.value() : grid_x_min = grid_x_min;
      gy.value() < grid_y_min ? grid_y_min = gy.value() : grid_y_min = grid_y_min;
      image_vec.emplace_back(fname, gx.value(), gy.value(), gc.value());   
    }
  }
  PLOG_INFO << "Total images found: " << image_vec.size() <<std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();
  auto [x_dim, y_dim, c_dim, num_dims] = GetZarrParams(v);

  if (image_vec.size() != 0){
    //std::list<tensorstore::WriteFutures> pending_writes;
    size_t write_failed_count = 0;
    std::string sample_tiff_file = image_vec[0].file_name;
    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto test_source, tensorstore::Open(
                    GetOmeTiffSpecToRead(sample_tiff_file),
                    tensorstore::OpenMode::open,
                    tensorstore::ReadWriteMode::read).result());
    auto test_image_shape = test_source.domain().shape();
    whole_image._chunk_size_x = test_image_shape[4];
    whole_image._chunk_size_y = test_image_shape[3];
    whole_image._full_image_width = (grid_x_max-grid_x_min+1)*whole_image._chunk_size_x;
    whole_image._full_image_height = (grid_y_max-grid_y_min+1)*whole_image._chunk_size_y;
    whole_image._num_channels = grid_c_max-grid_c_min+1;
    
    std::vector<std::int64_t> new_image_shape(num_dims,1);
    std::vector<std::int64_t> chunk_shape(num_dims,1);
    new_image_shape[y_dim] = whole_image._full_image_height;
    new_image_shape[x_dim] = whole_image._full_image_width;
    chunk_shape[y_dim] = whole_image._chunk_size_y;
    chunk_shape[x_dim] = whole_image._chunk_size_x;
    whole_image._data_type = test_source.dtype().name();
    if (v == VisType::NG_Zarr || v == VisType::Viv){
      new_image_shape[c_dim] = whole_image._num_channels;
    }

    auto output_spec = [&](){
      if (v == VisType::NG_Zarr || v == VisType::Viv){
        return GetZarrSpecToWrite(output_file + "/" + scale_key, new_image_shape, chunk_shape, ChooseBaseDType(test_source.dtype()).value().encoded_dtype);
      }  else if (v == VisType::PCNG){
        return GetNPCSpecToWrite(output_file, scale_key, new_image_shape, chunk_shape, 1, whole_image._num_channels, test_source.dtype().name(), true);
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
    for(const auto& i: image_vec){        
      th_pool.detach_task([&dest, i, x_dim=x_dim, y_dim=y_dim, c_dim=c_dim, v, &whole_image, grid_c_min, grid_x_min, grid_y_min]() {


        TENSORSTORE_CHECK_OK_AND_ASSIGN(auto source, tensorstore::Open(
                                    GetOmeTiffSpecToRead(i.file_name),
                                    tensorstore::OpenMode::open,
                                    tensorstore::ReadWriteMode::read).result());
        PLOG_INFO << "Opening "<< i.file_name;
        auto image_shape = source.domain().shape();
        auto image_width = image_shape[4];
        auto image_height = image_shape[3];
        auto array = tensorstore::AllocateArray({image_height, image_width},tensorstore::c_order,
                                                          tensorstore::value_init, source.dtype());

        // initiate a read
        tensorstore::Read(source | 
              tensorstore::Dims(3).ClosedInterval(0, image_height-1) |
              tensorstore::Dims(4).ClosedInterval(0, image_width-1) ,
              array).value();

        tensorstore::IndexTransform<> transform = tensorstore::IdentityTransform(dest.domain());
        if(v == VisType::PCNG){
          transform = (std::move(transform) | tensorstore::Dims("z", "channel").IndexSlice({0, i._c_grid-grid_c_min}) 
                                            | tensorstore::Dims(y_dim).SizedInterval((i._y_grid-grid_y_min)*whole_image._chunk_size_y, image_height) 
                                            | tensorstore::Dims(x_dim).SizedInterval((i._x_grid-grid_x_min)*whole_image._chunk_size_x, image_width)
                                            | tensorstore::Dims(x_dim, y_dim).Transpose({y_dim, x_dim})).value();

        } else if (v == VisType::NG_Zarr){
          transform = (std::move(transform) | tensorstore::Dims(c_dim).SizedInterval(i._c_grid-grid_c_min, 1) 
                                            | tensorstore::Dims(y_dim).SizedInterval((i._y_grid-grid_y_min)*whole_image._chunk_size_y, image_height) 
                                            | tensorstore::Dims(x_dim).SizedInterval((i._x_grid-grid_x_min)*whole_image._chunk_size_x, image_width)).value();
        } else if (v == VisType::Viv){
          transform = (std::move(transform) | tensorstore::Dims(c_dim).SizedInterval(i._c_grid-grid_c_min, 1) 
                                            | tensorstore::Dims(y_dim).SizedInterval((i._y_grid-grid_y_min)*whole_image._chunk_size_y, image_height) 
                                            | tensorstore::Dims(x_dim).SizedInterval((i._x_grid-grid_x_min)*whole_image._chunk_size_x, image_width)).value();
        }
        tensorstore::Write(array, dest | transform).value();
      });
    }

    th_pool.wait();
  }
  return std::move(whole_image);
}
} // ns argolid