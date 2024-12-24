#pragma once

#include <string>
#include <cmath>
#include <memory>
#include "ome_tiff_to_chunked_converter.h"
#include "chunked_pyramid_assembler.h"
#include "chunked_base_to_pyr_gen.h"
#include "../utilities/utilities.h"
#include "BS_thread_pool.hpp"
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
namespace argolid {
class OmeTiffToChunkedPyramid{
public:
    OmeTiffToChunkedPyramid(){
        auto log_file_name = "argolid_" + argolid::GetUTCString() + ".log";
        plog::init(plog::none, log_file_name.c_str());

    }
    void GenerateFromSingleFile(const std::string& input_file, const std::string& output_dir, 
                                int min_dim, VisType v, std::unordered_map<std::int64_t, DSType>& channel_ds_config);
    void GenerateFromCollection(const std::string& collection_path, const std::string& stitch_vector_file,
                                const std::string& image_name, const std::string& output_dir, 
                                int min_dim, VisType v, std::unordered_map<std::int64_t, DSType>& channel_ds_config);
    void SetLogLevel(int level){
        if (level>=0 && level<=6) {
            plog::get()->setMaxSeverity(plog::Severity(level));
        }
    }

private:
    OmeTiffToChunkedConverter _tiff_to_chunk;
    ChunkedBaseToPyramid _base_to_pyramid;
    OmeTiffCollToChunked _tiff_coll_to_chunk;
    BS::thread_pool<BS::tp::none> _th_pool;
};
} // ns argolid