#include <string_view>
#include <string>
#include <tuple>
#include <unordered_map>
#include <optional>
#include "../utilities/utilities.h"
#include "BS_thread_pool.hpp"
namespace argolid{

using image_map = std::unordered_map<std::string, std::tuple<std::uint32_t,uint32_t,uint32_t>>;

class PyramidView{
public:
    PyramidView(std::string_view image_path, 
                std::string_view pyramid_zarr_loc,
                std::string_view output_image_name,
                std::uint16_t x_spacing,
                std::uint16_t y_spacing):
        image_coll_path(image_path), 
        pyramid_zarr_path(pyramid_zarr_loc),
        image_name(output_image_name),
        x_spacing(x_spacing),
        y_spacing(y_spacing)
        {}
    
    void AssembleBaseLevel(VisType v, const image_map& map, const std::string& zarr_array_path);
    void GeneratePyramid(const image_map& map, 
                                    VisType v, 
                                    int min_dim,  
                                    const std::unordered_map<std::int64_t, DSType>& channel_ds_config);


private:
    std::string image_coll_path, pyramid_zarr_path, image_name;
    std::uint16_t x_spacing, y_spacing;
    BS::thread_pool<BS::tp::none> th_pool;
    ImageInfo base_image;
};
}


