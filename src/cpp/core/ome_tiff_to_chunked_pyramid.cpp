#include "ome_tiff_to_chunked_pyramid.h"
#include <filesystem>

namespace fs = std::filesystem;

namespace argolid {
void OmeTiffToChunkedPyramid::GenerateFromSingleFile(  const std::string& input_file,
                                                    const std::string& output_dir, 
                                                    int min_dim, VisType v, std::unordered_map<std::int64_t, DSType>& channel_ds_config){
    auto tiff_dims = GetTiffDims(input_file);
    if (tiff_dims.has_value()) {
        auto[image_height, image_width] = tiff_dims.value();
        int max_level = static_cast<int>(ceil(log2(std::max({image_width, image_height}))));
        int min_level = static_cast<int>(ceil(log2(min_dim)));   
        std::string tiff_file_name = fs::path(input_file).stem().string();
        std::string chunked_file_dir = output_dir + "/" + tiff_file_name + ".zarr";
        if (v == VisType::Viv){
            chunked_file_dir = chunked_file_dir + "/data.zarr/0";
        }

        int base_level_key = 0;
        auto max_level_key = max_level-min_level+1+base_level_key;
        PLOG_INFO << "Converting base image...";
        _tiff_to_chunk.Convert(input_file, chunked_file_dir, std::to_string(base_level_key), v, _th_pool);
        PLOG_INFO << "Generating image pyramids...";
        _base_to_pyramid.CreatePyramidImages(chunked_file_dir, chunked_file_dir, 0, min_dim, v, channel_ds_config, _th_pool);
        PLOG_INFO << "Writing metadata...";
        WriteMultiscaleMetadataForSingleFile(input_file, output_dir, base_level_key, max_level_key, v);

    }
}


void OmeTiffToChunkedPyramid::GenerateFromCollection(
                const std::string& collection_path, 
                const std::string& stitch_vector_file,
                const std::string& image_name,
                const std::string& output_dir, 
                int min_dim, 
                VisType v,
                std::unordered_map<std::int64_t, DSType>& channel_ds_config){
    std::string chunked_file_dir = output_dir + "/" + image_name + ".zarr";
    if (v == VisType::Viv){
        chunked_file_dir = chunked_file_dir + "/data.zarr/0";
    }

    int base_level_key = 0;
    PLOG_INFO << "Assembling base image...";
    auto whole_image =_tiff_coll_to_chunk.Assemble(collection_path, stitch_vector_file, chunked_file_dir, std::to_string(base_level_key), v, _th_pool);
    int max_level = static_cast<int>(ceil(log2(std::max({whole_image._full_image_width, whole_image._full_image_width}))));
    int min_level = static_cast<int>(ceil(log2(min_dim)));
    auto max_level_key = max_level-min_level+1+base_level_key;
    PLOG_INFO << "Generating image pyramids...";
    _base_to_pyramid.CreatePyramidImages(chunked_file_dir, chunked_file_dir,base_level_key, min_dim, v, channel_ds_config, _th_pool);
    PLOG_INFO << "Writing metadata...";
    WriteMultiscaleMetadataForImageCollection(image_name, output_dir, base_level_key, max_level_key, v, whole_image);
}
} // ns argolid