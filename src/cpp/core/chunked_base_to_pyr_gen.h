#pragma once
#include <string>
#include <unordered_map>
#include "BS_thread_pool.hpp"
#include "../utilities/utilities.h"
namespace argolid{
class ChunkedBaseToPyramid{
public:
    ChunkedBaseToPyramid() = default;
    void CreatePyramidImages(   const std::string& input_chunked_dir,
                                const std::string& output_root_dir, 
                                int base_scale_key,
                                int min_dim, 
                                VisType v, 
                                const std::unordered_map<std::int64_t, DSType>& channel_ds_config,
                                BS::thread_pool<BS::tp::none>& th_pool);

private:
    template<typename T>
    void WriteDownsampledImage( const std::string& input_file, const std::string& input_scale_key, 
                                const std::string& output_file, const std::string& output_scale_key,
                                int resolution, VisType v,
                                const std::unordered_map<std::int64_t, DSType>& channel_ds_config, 
                                BS::thread_pool<BS::tp::none>& th_pool);
};
} // ns argolid
