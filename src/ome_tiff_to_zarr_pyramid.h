#pragma once

#include <string>
#include <tiffio.h>
#include <cmath>
#include <memory>
#include "ome_tiff_to_zarr_converter.h"
#include "zarr_pyramid_assembler.h"
#include "zarr_base_to_pyr_gen.h"
#include "utilities.h"
#include "BS_thread_pool.hpp"
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

class OmeTiffToChunkedPyramid{
public:
    OmeTiffToChunkedPyramid(){
        auto log_file_name = ::GetUTCString() + ".txt";
        plog::init(plog::none, log_file_name.c_str());

    }
    void GenerateFromSingleFile(const std::string& input_file, const std::string& output_dir, 
                                int min_dim, VisType v, std::unordered_map<std::int64_t, DSType>& channel_ds_config);
    void GenerateFromCollection(const std::string& collection_path, const std::string& stitch_vector_file,
                                const std::string& image_name, const std::string& output_dir, 
                                int min_dim, VisType v, std::unordered_map<std::int64_t, DSType>& channel_ds_config);
    void SetLogLevel(int level){
        switch (level)
        {
        case 0:
            plog::get()->setMaxSeverity(plog::none);
            break;
        case 1:
            plog::get()->setMaxSeverity(plog::fatal);
            break;
        case 2:
            plog::get()->setMaxSeverity(plog::error);
            break;       
        case 3:
            plog::get()->setMaxSeverity(plog::warning);
            break;
        case 4:
            plog::get()->setMaxSeverity(plog::info);
            break;
        case 5:
            plog::get()->setMaxSeverity(plog::debug);
            break;
        case 6:
            plog::get()->setMaxSeverity(plog::verbose);
            break;
        default:
            plog::get()->setMaxSeverity(plog::none);
            break;
        }
        
            
    }

private:
    std::unique_ptr<OmeTiffToZarrConverter> _zpw_ptr = nullptr;
    std::unique_ptr<ChunkedBaseToPyramid> _zpg_ptr = nullptr;
    std::unique_ptr<OmeTiffCollToChunked> _tiff_coll_to_zarr_ptr = nullptr;
    BS::thread_pool _th_pool;

    void WriteMultiscaleMetadataForImageCollection( const std::string& input_file, 
                                                    const std::string& output_dir,
                                                    int min_level, int max_level,
                                                    VisType v);
    void WriteMultiscaleMetadataForSingleFile(  const std::string& input_file, 
                                                const std::string& output_dir,
                                                int min_level, int max_level,
                                                VisType v);
    void ExtractAndWriteXML(const std::string& input_file, const std::string& xml_loc);
    void WriteTSZattrFile(const std::string& tiff_file_name, const std::string& zattr_file_loc, int min_level, int max_level);
    void WriteVivZattrFile(const std::string& tiff_file_name, const std::string& zattr_file_loc, int min_level, int max_level);
    void WriteVivZgroupFiles(const std::string& output_dir);
     
};