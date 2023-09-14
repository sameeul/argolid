#pragma once

#include <string>
#include <tiffio.h>
#include <cmath>
#include <memory>
#include "ome_tiff_to_chunked_converter.h"
#include "chunked_pyramid_assembler.h"
#include "chunked_base_to_pyr_gen.h"
#include "utilities.h"
#include "BS_thread_pool.hpp"
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

class OmeTiffToChunkedPyramid{
public:
    OmeTiffToChunkedPyramid(){
        auto log_file_name = "argolid_" + ::GetUTCString() + ".log";
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
    BS::thread_pool _th_pool;

    void WriteMultiscaleMetadataForImageCollection( const std::string& input_file, 
                                                    const std::string& output_dir,
                                                    int min_level, int max_level,
                                                    VisType v, ImageInfo& whole_image);
    void WriteMultiscaleMetadataForSingleFile(  const std::string& input_file, 
                                                const std::string& output_dir,
                                                int min_level, int max_level,
                                                VisType v);
    void ExtractAndWriteXML(const std::string& input_file, const std::string& xml_loc);
    void WriteTSZattrFile(const std::string& tiff_file_name, const std::string& zattr_file_loc, int min_level, int max_level);
    void WriteVivZattrFile(const std::string& tiff_file_name, const std::string& zattr_file_loc, int min_level, int max_level);
    void WriteVivZgroupFiles(const std::string& output_dir);
     
};