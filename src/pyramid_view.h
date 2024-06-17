#include <string_view>
#include <string>
#include <tuple>
#include <unordered_map>
#include <optional>
#include "utilities.h"
#include "BS_thread_pool.hpp"

namespace argolid{

using image_map = std::unordered_map<std::string, std::tuple<std::uint32_t,uint32_t,uint32_t>>;

class PyramidView{
public:
    PyramidView(std::string_view image_path, 
                std::string base_zarr_loc,
                std::string pyramid_zarr_loc,
                std::string output_image_name,
                image_map& map):
        image_coll_path(image_path), 
        base_zarr_path(base_zarr_loc),
        pyramid_zarr_path(pyramid_zarr_loc),
        image_name(output_image_name),
        base_image_map(map)
        {}
    
    void AssembleBaseLevel(VisType v);
    void AssembleBaseLevel(VisType v, image_map m, const std::string& output_path);
    void GeneratePyramid(std::optional<image_map> map, 
                                    VisType v, 
                                    int min_dim,  
                                    std::unordered_map<std::int64_t, DSType>& channel_ds_config);


private:
    std::string image_coll_path, base_zarr_path, pyramid_zarr_path, image_name, pyramid_root;
    std::uint16_t max_level;
    image_map base_image_map;
    BS::thread_pool th_pool;
    ImageInfo base_image;
};
}


