#include "zarr_base_to_pyr_gen.h"
#include "downsample.h"
#include "utilities.h"
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include <string>
#include <stdint.h>
#include<typeinfo>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>
#include <chrono>
#include <future>
#include <cmath>

#include "tensorstore/tensorstore.h"
#include "tensorstore/context.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/zarr/dtype.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/open.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <filesystem>
namespace fs = std::filesystem;

using ::tensorstore::Context;
using ::tensorstore::internal_zarr::ChooseBaseDType;
using namespace std::chrono_literals;

void ChunkedBaseToPyramid::CreatePyramidImages( const std::string& input_zarr_dir,
                                                const std::string& output_root_dir, 
                                                int base_level_key,
                                                int min_dim, 
                                                VisType v, 
                                                std::unordered_map<std::int64_t, DSType>& channel_ds_config,
                                                BS::thread_pool& th_pool)
{
    int resolution = 1; // this gets doubled in each level up
    tensorstore::Spec input_spec{};
    if (v == VisType::NG_Zarr | v == VisType::Viv){
      input_spec = GetZarrSpecToRead(input_zarr_dir, std::to_string(base_level_key));
    } else if (v == VisType::PCNG){
      input_spec = GetNPCSpecToRead(input_zarr_dir, std::to_string(base_level_key));
    }

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto test_store, tensorstore::Open(
                            input_spec,
                            tensorstore::OpenMode::open,
                            tensorstore::ReadWriteMode::read).result());
    auto data_type = GetDataTypeCode(test_store.dtype().name());
    int x_ind, y_ind;
    auto shape = test_store.domain().shape();

    if (v == VisType::Viv){ //5D file
        x_ind = 4;
        y_ind = 3;
    
    } else if (v == VisType::NG_Zarr){ // 3D file
        x_ind = 3;
        y_ind = 2;
    
    } else if (v == VisType::PCNG ){ // 3D file
        x_ind = 1;
        y_ind = 0;
    }
    auto max_level = static_cast<int>(ceil(log2(std::max(shape[x_ind], shape[y_ind]))));
    auto min_level = static_cast<int>(ceil(log2(min_dim)));
    auto max_key = max_level-min_level+1+base_level_key;

    for (int i=base_level_key; i<max_key; ++i){
        resolution *= 2;
        switch(data_type){
          case 1:
            WriteDownsampledImage<uint8_t>(input_zarr_dir, std::to_string(i), output_root_dir, std::to_string(i+1), resolution, v, channel_ds_config, th_pool);
            break;
          case 2:
            WriteDownsampledImage<uint16_t>(input_zarr_dir, std::to_string(i), output_root_dir, std::to_string(i+1), resolution, v, channel_ds_config, th_pool);
            break;
          case 4:
            WriteDownsampledImage<uint32_t>(input_zarr_dir, std::to_string(i), output_root_dir, std::to_string(i+1), resolution, v, channel_ds_config, th_pool);
            break;
          case 8:
            WriteDownsampledImage<uint64_t>(input_zarr_dir, std::to_string(i), output_root_dir, std::to_string(i+1), resolution, v, channel_ds_config, th_pool);
            break;
          case 16:
            WriteDownsampledImage<int8_t>(input_zarr_dir, std::to_string(i), output_root_dir, std::to_string(i+1), resolution, v, channel_ds_config, th_pool);
            break;
          case 32:
            WriteDownsampledImage<int16_t>(input_zarr_dir, std::to_string(i), output_root_dir, std::to_string(i+1), resolution, v, channel_ds_config, th_pool);
            break;
          case 64:
            WriteDownsampledImage<int32_t>(input_zarr_dir, std::to_string(i), output_root_dir, std::to_string(i+1), resolution, v, channel_ds_config, th_pool);
            break;
          case 128:
            WriteDownsampledImage<int64_t>(input_zarr_dir, std::to_string(i), output_root_dir, std::to_string(i+1), resolution, v, channel_ds_config, th_pool);
            break;
          case 256:
            WriteDownsampledImage<float>(input_zarr_dir, std::to_string(i), output_root_dir, std::to_string(i+1), resolution, v, channel_ds_config, th_pool);
            break;
          case 512:
            WriteDownsampledImage<double>(input_zarr_dir, std::to_string(i), output_root_dir, std::to_string(i+1), resolution, v, channel_ds_config, th_pool);
            break;
          default:
            break;
          }
    } 
}

template <typename T>
void ChunkedBaseToPyramid::WriteDownsampledImage(   const std::string& input_file, const std::string& input_scale_key, 
                                                    const std::string& output_file, const std::string& output_scale_key,
                                                    int resolution, VisType v, 
                                                    std::unordered_map<std::int64_t, DSType>& channel_ds_config,
                                                    BS::thread_pool& th_pool)
{
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
        x_dim = 1;
        y_dim = 0;
        c_dim = 3;
        num_dims = 3;
    }
    tensorstore::Spec input_spec{};
    if (v == VisType::NG_Zarr | v == VisType::Viv){
      input_spec = GetZarrSpecToRead(input_file, input_scale_key);
    } else if (v == VisType::PCNG){
      input_spec = GetNPCSpecToRead(input_file, input_scale_key);
    }
    //tensorstore::Context context = Context::Default();
    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store1, tensorstore::Open(
                            input_spec,
                            tensorstore::OpenMode::open,
                            tensorstore::ReadWriteMode::read).result());
    auto prev_image_shape = store1.domain().shape();
    auto read_chunk_shape = store1.chunk_layout().value().read_chunk_shape();

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto base_zarr_dtype,
                                        ChooseBaseDType(store1.dtype()));
    auto prev_x_max = static_cast<std::int64_t>(prev_image_shape[x_dim]);
    auto prev_y_max = static_cast<std::int64_t>(prev_image_shape[y_dim]);

    auto cur_x_max = static_cast<std::int64_t>(ceil(prev_x_max/2.0));
    auto cur_y_max = static_cast<std::int64_t>(ceil(prev_y_max/2.0));
    auto num_channels = static_cast<std::int64_t>(prev_image_shape[c_dim]);
    //std::int64_t num_channels = 1;
    std::vector<std::int64_t> new_image_shape(num_dims,1);
    std::vector<std::int64_t> chunk_shape(num_dims,1);

    new_image_shape[y_dim] = cur_y_max;
    new_image_shape[x_dim] = cur_x_max;

    chunk_shape[y_dim] = static_cast<std::int64_t>(read_chunk_shape[y_dim]);
    chunk_shape[x_dim] = static_cast<std::int64_t>(read_chunk_shape[x_dim]);

    auto num_rows = static_cast<std::int64_t>(ceil(1.0*cur_y_max/chunk_shape[y_dim]));
    auto num_cols = static_cast<std::int64_t>(ceil(1.0*cur_x_max/chunk_shape[x_dim]));

    tensorstore::Spec output_spec{};
    auto open_mode = tensorstore::OpenMode::create;

    if (v == VisType::NG_Zarr | v == VisType::Viv){
      new_image_shape[c_dim] = prev_image_shape[c_dim];
      output_spec = GetZarrSpecToWrite(output_file + "/" + output_scale_key, new_image_shape, chunk_shape, base_zarr_dtype.encoded_dtype);
      open_mode = open_mode | tensorstore::OpenMode::delete_existing;
    } else if (v == VisType::PCNG){
      output_spec = GetNPCSpecToWrite(output_file, output_scale_key, new_image_shape, chunk_shape, resolution, num_channels, store1.dtype().name(), false);
    }
    
    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store2, tensorstore::Open(
                            output_spec,
                            open_mode,
                            tensorstore::ReadWriteMode::write).result());

    std::unique_ptr<std::vector<T>> (*downsampling_func_ptr)(std::vector<T>&, std::int64_t, std::int64_t);   
    downsampling_func_ptr = &DownsampleAverage; // default
    for(std::int64_t c=0; c<num_channels; ++c){
        auto it = channel_ds_config.find(c);
        if (it != channel_ds_config.end()){
            
            if (it->second == DSType::Mode_Max){
                downsampling_func_ptr = &DownsampleModeMax;
                PLOG_DEBUG<< "Channel ID " << it->first <<" Downsampling method Mode Max";
            } else if (it->second == DSType::Mode_Min){
                downsampling_func_ptr = &DownsampleModeMin;
                PLOG_DEBUG<< "Channel ID " << it->first <<" Downsampling method Mode Min";
            } else if (it->second == DSType::Mean){
                PLOG_DEBUG<< "Channel ID " << it->first <<" Downsampling method Mean";
                downsampling_func_ptr = &DownsampleAverage;
            } 
        }
        for(std::int64_t i=0; i<num_rows; ++i){
            auto y_start = i*chunk_shape[y_dim];
            auto y_end = std::min({(i+1)*chunk_shape[y_dim], cur_y_max});

            auto prev_y_start = 2*y_start;
            auto prev_y_end = std::min({2*y_end, prev_y_max});
            for(std::int64_t j=0; j<num_cols; ++j){
                auto x_start = j*chunk_shape[x_dim];
                auto x_end = std::min({(j+1)*chunk_shape[x_dim], cur_x_max});
                auto prev_x_start = 2*x_start;
                auto prev_x_end = std::min({2*x_end, prev_x_max});
                th_pool.push_task([ &store1, &store2, 
                                    prev_x_start, prev_x_end, prev_y_start, prev_y_end, 
                                    x_start, x_end, y_start, y_end, 
                                    x_dim, y_dim, c_dim, c, v, downsampling_func_ptr](){  
                    std::vector<T> read_buffer((prev_x_end-prev_x_start)*(prev_y_end-prev_y_start));
                    auto array = tensorstore::Array(read_buffer.data(), {prev_y_end-prev_y_start, prev_x_end-prev_x_start}, tensorstore::c_order);

                    tensorstore::IndexTransform<> input_transform = tensorstore::IdentityTransform(store1.domain());
                    
                    if(v == VisType::PCNG){
                      input_transform = (std::move(input_transform) | tensorstore::Dims(2, 3).IndexSlice({0,c})).value();
                    } else 
                    if (v == VisType::Viv || v == VisType::NG_Zarr){
                      input_transform = (std::move(input_transform) | tensorstore::Dims(c_dim).SizedInterval(c,1)).value();
                    } 

                    input_transform = (std::move(input_transform) | tensorstore::Dims(y_dim).ClosedInterval(prev_y_start, prev_y_end-1) 
                                                        | tensorstore::Dims(x_dim).ClosedInterval(prev_x_start, prev_x_end-1)).value(); 

                    tensorstore::Read(store1 | input_transform, tensorstore::UnownedToShared(array)).value();

                    auto result = downsampling_func_ptr(read_buffer, (prev_y_end-prev_y_start), (prev_x_end-prev_x_start));
                    auto result_array = tensorstore::Array(result->data(), {y_end-y_start, x_end-x_start}, tensorstore::c_order);


                    tensorstore::IndexTransform<> output_transform = tensorstore::IdentityTransform(store2.domain());
                    if(v == VisType::PCNG){
                      output_transform = (std::move(output_transform) | tensorstore::Dims(2, 3).IndexSlice({0,c})).value();
                    } else
                    if (v == VisType::Viv || v == VisType::NG_Zarr){
                      output_transform = (std::move(output_transform) | tensorstore::Dims(c_dim).SizedInterval(c,1)).value();
                    } 
                    output_transform = (std::move(output_transform) | tensorstore::Dims(y_dim).ClosedInterval(y_start, y_end-1) 
                                                                    | tensorstore::Dims(x_dim).ClosedInterval(x_start, x_end-1)).value(); 

                    tensorstore::Write(tensorstore::UnownedToShared(result_array), store2 | output_transform).value();  
                }); 
            }
        }
       
    }
    th_pool.wait_for_tasks();
}