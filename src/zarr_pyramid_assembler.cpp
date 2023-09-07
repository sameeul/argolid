#include "zarr_pyramid_assembler.h"
#include "utilities.h"
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

#include "tensorstore/tensorstore.h"
#include "tensorstore/context.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/zarr/dtype.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/open.h"

#include "filepattern.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <filesystem>
namespace fs = std::filesystem;

using ::tensorstore::Context;
using ::tensorstore::internal_zarr::ChooseBaseDType;

std::int64_t retrieve_val(const std::string& var, const Map& m){
    auto it = m.find(var);
    if(it != m.end()){
        return static_cast<std::int64_t>(std::get<int>(it->second));
    } else {
        return 0;
    }
}



void OmeTiffCollToChunked::Assemble(const std::string& input_dir,
                                    const std::string& pattern ,
                                    const std::string& output_file, 
                                    const std::string& scale_key, 
                                    VisType v, 
                                    BS::thread_pool& th_pool)
{
  int grid_x_max = 0, grid_y_max = 0, grid_c_max = 0;
  std::vector<ImageSegment> image_vec;
  _chunk_size_x = 0;
  _chunk_size_y = 0;
  _full_image_width = 0;
  _full_image_height = 0;

  auto fp = std::make_unique<FilePattern> (input_dir, pattern);
  auto files = fp->getFiles();

  for(const auto& [map, values]: files){
    auto gc = retrieve_val("c", map);
    auto gx = retrieve_val("x", map);
    auto gy = retrieve_val("y", map);
    auto fname = values[0].string();  
    gc > grid_c_max ? grid_c_max = gc : grid_c_max = grid_c_max;
    gx > grid_x_max ? grid_x_max = gx : grid_x_max = grid_x_max;
    gy > grid_y_max ? grid_y_max = gy : grid_y_max = grid_y_max;
    image_vec.emplace_back(fname, gx, gy, gc);   
  }
  PLOG_INFO << "Total images found: " << image_vec.size() <<std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();
  int num_dims, x_dim, y_dim, c_dim;

  if (v == VisType::Viv){ //5D file
    x_dim = 4;
    y_dim = 3;
    c_dim = 1;
    num_dims = 5;
  
  } else if (v == VisType::NG_Zarr){ // 3D file
    x_dim = 3;
    y_dim = 2;
    c_dim = 0;
    num_dims = 4;
  
  } else if (v == VisType::PCNG ){ // 3D file
    x_dim = 0;
    y_dim = 1;
    c_dim = 3;
    num_dims = 3;
  }

  if (image_vec.size() != 0){
    //std::list<tensorstore::WriteFutures> pending_writes;
    size_t write_failed_count = 0;
    std::string sample_tiff_file = image_vec[0].file_name;
    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto test_source, tensorstore::Open(
                    GetOmeTiffSpecToRead(sample_tiff_file),
                    tensorstore::OpenMode::open,
                    tensorstore::ReadWriteMode::read).result());
    auto test_image_shape = test_source.domain().shape();
    _chunk_size_x = test_image_shape[4];
    _chunk_size_y = test_image_shape[3];
    _full_image_width = (grid_x_max+1)*_chunk_size_x;
    _full_image_height = (grid_y_max+1)*_chunk_size_y;
    _num_channels = grid_c_max+1;

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto base_zarr_dtype,
                                        ChooseBaseDType(test_source.dtype()));
    
    tensorstore::Spec output_spec{};
    
    std::vector<std::int64_t> new_image_shape(num_dims,1);
    std::vector<std::int64_t> chunk_shape(num_dims,1);
    new_image_shape[y_dim] = _full_image_height;
    new_image_shape[x_dim] = _full_image_width;
    chunk_shape[y_dim] = _chunk_size_y;
    chunk_shape[x_dim] = _chunk_size_x;
    _data_type_string = test_source.dtype().name();
    if (v == VisType::NG_Zarr || v == VisType::Viv){
      new_image_shape[c_dim] = _num_channels;
      output_spec = GetZarrSpecToWrite(output_file + "/" + scale_key, new_image_shape, chunk_shape, base_zarr_dtype.encoded_dtype);
    }  else if (v == VisType::PCNG){
      output_spec = GetNPCSpecToWrite(output_file, scale_key, new_image_shape, chunk_shape, 1, _num_channels, test_source.dtype().name(), true);
    }

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto dest, tensorstore::Open(
                                output_spec,
                                tensorstore::OpenMode::create |
                                tensorstore::OpenMode::delete_existing,
                                tensorstore::ReadWriteMode::write).result());
    
    auto t4 = std::chrono::high_resolution_clock::now();
    for(const auto& i: image_vec){        
      th_pool.push_task([&dest, i, x_dim, y_dim, c_dim, this, v](){


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
          transform = (std::move(transform) | tensorstore::Dims("z", "channel").IndexSlice({0, i._c_grid}) 
                                            | tensorstore::Dims(y_dim).SizedInterval(i._y_grid*this->_chunk_size_y, image_height) 
                                            | tensorstore::Dims(x_dim).SizedInterval(i._x_grid*this->_chunk_size_x, image_width)
                                            | tensorstore::Dims(x_dim, y_dim).Transpose({y_dim, x_dim})).value();

        } else if (v == VisType::NG_Zarr){
          transform = (std::move(transform) | tensorstore::Dims(c_dim).SizedInterval(i._c_grid, 1) 
                                            | tensorstore::Dims(y_dim).SizedInterval(i._y_grid*this->_chunk_size_y, image_height) 
                                            | tensorstore::Dims(x_dim).SizedInterval(i._x_grid*this->_chunk_size_x, image_width)).value();
        } else if (v == VisType::Viv){
          transform = (std::move(transform) | tensorstore::Dims(c_dim).SizedInterval(i._c_grid, 1) 
                                            | tensorstore::Dims(y_dim).SizedInterval(i._y_grid*this->_chunk_size_y, image_height) 
                                            | tensorstore::Dims(x_dim).SizedInterval(i._x_grid*this->_chunk_size_x, image_width)).value();
        }
        tensorstore::Write(array, dest | transform).value();
      });
    }

    th_pool.wait_for_tasks();
  }
}

void OmeTiffCollToChunked::GenerateOmeXML(const std::string& image_name, const std::string& output_file){

    pugi::xml_document doc;

    // Create the root element <OME>
    pugi::xml_node omeNode = doc.append_child("OME");
    
    // Add the namespaces and attributes to the root element
    omeNode.append_attribute("xmlns") = "http://www.openmicroscopy.org/Schemas/OME/2016-06";
    omeNode.append_attribute("xmlns:xsi") = "http://www.w3.org/2001/XMLSchema-instance";
    omeNode.append_attribute("Creator") = "OmePyrGen 0.0.1";
    omeNode.append_attribute("UUID") = "urn:uuid:ce3367ae-0512-4e87-a045-20d87db14001";
    omeNode.append_attribute("xsi:schemaLocation") = "http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd";

    // Create the <Image> element
    pugi::xml_node imageNode = omeNode.append_child("Image");
    imageNode.append_attribute("ID") = image_name.c_str();;
    imageNode.append_attribute("Name") =image_name.c_str();

    // Create the <Pixels> element
    pugi::xml_node pixelsNode = imageNode.append_child("Pixels");
    pixelsNode.append_attribute("BigEndian") = "false";
    pixelsNode.append_attribute("DimensionOrder") = "XYZCT";
    pixelsNode.append_attribute("ID") = "Pixels:0";
    pixelsNode.append_attribute("Interleaved") = "false";
    pixelsNode.append_attribute("SizeC") = std::to_string(_num_channels).c_str();;
    pixelsNode.append_attribute("SizeT") = "1";
    pixelsNode.append_attribute("SizeX") = std::to_string(_full_image_width).c_str();
    pixelsNode.append_attribute("SizeY") = std::to_string(_full_image_height).c_str();
    pixelsNode.append_attribute("SizeZ") = "1";
    pixelsNode.append_attribute("Type") = _data_type_string.c_str();

    // Create the <Channel> elements
    for(std::int64_t i=0; i<_num_channels; ++i){
      pugi::xml_node channelNode = pixelsNode.append_child("Channel");
      channelNode.append_attribute("ID") = ("Channel:0:" + std::to_string(i)).c_str();
      channelNode.append_attribute("SamplesPerPixel") = "1";
      // Create the <LightPath> elements
      channelNode.append_child("LightPath");
    }
  
    doc.save_file(output_file.c_str());
}