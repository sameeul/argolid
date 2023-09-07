#include "ome_tiff_to_zarr_pyramid.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

void OmeTiffToChunkedPyramid::GenerateFromSingleFile(  const std::string& input_file,
                                                    const std::string& output_dir, 
                                                    int min_dim, VisType v, std::unordered_map<std::int64_t, DSType>& channel_ds_config){
    TIFF *tiff_ = TIFFOpen(input_file.c_str(), "r");
    if (tiff_ != nullptr) {
        uint32_t image_width = 0, image_height = 0;
        TIFFGetField(tiff_, TIFFTAG_IMAGEWIDTH, &image_width);
        TIFFGetField(tiff_, TIFFTAG_IMAGELENGTH, &image_height);
        int max_level = static_cast<int>(ceil(log2(std::max({image_width, image_height}))));
        int min_level = static_cast<int>(ceil(log2(min_dim)));
        TIFFClose(tiff_);     
        std::string tiff_file_name = fs::path(input_file).stem().string();
        std::string zarr_file_dir = output_dir + "/" + tiff_file_name + ".zarr";
        if (v == VisType::Viv){
            zarr_file_dir = zarr_file_dir + "/data.zarr/0";
        }

        int base_level_key = 0;
        auto max_level_key = max_level-min_level+1+base_level_key;
        _zpw_ptr = std::make_unique<OmeTiffToZarrConverter>();
        PLOG_INFO << "Converting base image...";
        _zpw_ptr->Convert(input_file, zarr_file_dir, std::to_string(base_level_key), v, _th_pool);
        PLOG_INFO << "Generating image pyramids...";
        _zpg_ptr = std::make_unique<ChunkedBaseToPyramid>();
        _zpg_ptr->CreatePyramidImages(zarr_file_dir, zarr_file_dir, 0, min_dim, v, channel_ds_config, _th_pool);
        PLOG_INFO << "Writing metadata...";
        WriteMultiscaleMetadataForSingleFile(input_file, output_dir, base_level_key, max_level_key, v);

    }
}

void OmeTiffToChunkedPyramid::WriteMultiscaleMetadataForSingleFile( const std::string& input_file , const std::string& output_dir, 
                                                                    int min_level, int max_level, VisType v)
{
    std::string tiff_file_name = fs::path(input_file).stem().string();
    std::string zarr_file_dir = output_dir + "/" + tiff_file_name + ".zarr";
    if(v == VisType::NG_Zarr){

        WriteTSZattrFile(tiff_file_name, zarr_file_dir, min_level, max_level);
    } else if (v == VisType::Viv){
        ExtractAndWriteXML(input_file, zarr_file_dir);
        WriteVivZattrFile(tiff_file_name, zarr_file_dir+"/data.zarr/0/", min_level, max_level);
        WriteVivZgroupFiles(zarr_file_dir);
    }
}


void OmeTiffToChunkedPyramid::WriteMultiscaleMetadataForImageCollection(const std::string& image_file_name , const std::string& output_dir, 
                                                                        int min_level, int max_level, VisType v)
{
    std::string zarr_file_dir = output_dir + "/" + image_file_name + ".zarr";
    if(v == VisType::NG_Zarr){
        WriteTSZattrFile(image_file_name, zarr_file_dir, min_level, max_level);
    } else if (v == VisType::Viv){
        _tiff_coll_to_zarr_ptr->GenerateOmeXML(image_file_name, zarr_file_dir+"/METADATA.ome.xml");                   
        WriteVivZattrFile(image_file_name, zarr_file_dir+"/data.zarr/0/", min_level, max_level);
        WriteVivZgroupFiles(zarr_file_dir);
    }
}

void OmeTiffToChunkedPyramid::WriteVivZgroupFiles(const std::string& output_loc){
    std::string zgroup_text = "{\"zarr_format\": 2}";
    std::ofstream zgroup_file_1(output_loc+"/data.zarr/.zgroup", std::ios_base::out );
    if(zgroup_file_1.is_open()){
        zgroup_file_1 << zgroup_text << std::endl;
    }
    std::ofstream zgroup_file_2( output_loc + "/data.zarr/0/.zgroup", std::ios_base::out );
    if(zgroup_file_2.is_open()){
        zgroup_file_2 << zgroup_text << std::endl;
    }
}


void OmeTiffToChunkedPyramid::ExtractAndWriteXML(const std::string& input_file, const std::string& xml_loc){
    TIFF *tiff_ = TIFFOpen(input_file.c_str(), "r");
    if (tiff_ != nullptr) {
        char* infobuf;
        TIFFGetField(tiff_, TIFFTAG_IMAGEDESCRIPTION , &infobuf);
        char* new_pos = strstr(infobuf, "<OME");
        std::ofstream metadata_file( xml_loc+"/METADATA.ome.xml", std::ios_base::out );
        if(metadata_file.is_open()){
            metadata_file << new_pos << std::endl;
        }
        if(!metadata_file){
            PLOG_INFO << "Unable to write metadata file";
        }
        TIFFClose(tiff_);
    }
}

void OmeTiffToChunkedPyramid::WriteTSZattrFile(const std::string& tiff_file_name, const std::string& zarr_root_dir, int min_level, int max_level){

    json zarr_multiscale_axes;
    zarr_multiscale_axes = json::parse(R"([
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"}
                ])");
    
    
    float level = 1.0;
    json scale_metadata_list = json::array();
    for(int i=min_level; i<=max_level; ++i){
        json scale_metadata;
        scale_metadata["path"] = std::to_string(i);
        scale_metadata["coordinateTransformations"] = {{{"type", "scale"}, {"scale", {1.0, 1.0, level, level}}}};
        scale_metadata_list.push_back(scale_metadata);
        level = level*2;
    }

    json combined_metadata;
    combined_metadata["datasets"] = scale_metadata_list;
    combined_metadata["version"] = "0.4";
    combined_metadata["axes"] = zarr_multiscale_axes;
    combined_metadata["name"] = tiff_file_name;
    combined_metadata["metadata"] = {{"method", "mean"}};
    json final_formated_metadata;
#ifdef _WIN32
    final_formated_metadata["multiscales"][0] = {combined_metadata};
#else
    final_formated_metadata["multiscales"] = {combined_metadata};
#endif
    std::ofstream f(zarr_root_dir + "/.zattrs",std::ios_base::trunc |std::ios_base::out);
    if (f.is_open()){
        f << final_formated_metadata;
    } else {
        PLOG_INFO <<"Unable to write .zattr file at "<< zarr_root_dir << ".";
    }
}

void OmeTiffToChunkedPyramid::WriteVivZattrFile(const std::string& tiff_file_name, const std::string& zattr_file_loc, int min_level, int max_level){
    
    json scale_metadata_list = json::array();
    for(int i=min_level; i<=max_level; ++i){
        json scale_metadata;
        scale_metadata["path"] = std::to_string(i);
        scale_metadata_list.push_back(scale_metadata);
    }

    json combined_metadata;
    combined_metadata["datasets"] = scale_metadata_list;
    combined_metadata["version"] = "0.1";
    combined_metadata["name"] = tiff_file_name;
    combined_metadata["metadata"] = {{"method", "mean"}};
    json final_formated_metadata;
#ifdef _WIN32
    final_formated_metadata["multiscales"][0] = {combined_metadata};
#else
    final_formated_metadata["multiscales"] = {combined_metadata};
#endif    
    std::ofstream f(zattr_file_loc + "/.zattrs",std::ios_base::trunc |std::ios_base::out);
    if (f.is_open()){   
        f << final_formated_metadata;
    } else {
        std::cout << "Unable to write .zattr file at " << zattr_file_loc << "." << std::endl;
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
    _tiff_coll_to_zarr_ptr = std::make_unique<OmeTiffCollToChunked>();  
    std::string zarr_file_dir = output_dir + "/" + image_name + ".zarr";
    if (v == VisType::Viv){
        zarr_file_dir = zarr_file_dir + "/data.zarr/0";
    }

    int base_level_key = 0;
    PLOG_INFO << "Assembling base image...";
    _tiff_coll_to_zarr_ptr->Assemble(collection_path, stitch_vector_file, zarr_file_dir, std::to_string(base_level_key), v, _th_pool);
    int max_level = static_cast<int>(ceil(log2(std::max({  _tiff_coll_to_zarr_ptr->image_width(), 
                                                        _tiff_coll_to_zarr_ptr->image_height()}))));
    int min_level = static_cast<int>(ceil(log2(min_dim)));
    auto max_level_key = max_level-min_level+1+base_level_key;
    PLOG_INFO << "Generating image pyramids...";
    _zpg_ptr = std::make_unique<ChunkedBaseToPyramid>();
    _zpg_ptr->CreatePyramidImages(zarr_file_dir, zarr_file_dir,base_level_key, min_dim, v, channel_ds_config, _th_pool);
    PLOG_INFO << "Writing metadata...";
    WriteMultiscaleMetadataForImageCollection(image_name, output_dir, base_level_key, max_level_key, v);
}