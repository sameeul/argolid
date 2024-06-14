#include <string_view>
#include <string>
#include <tuple>
#include <unordered_map>
#include <optional>

namespace argolid{

using image_map = std::unordered_map<std::string, std::tuple<std::uint32_t,uint32_t,uint32_t>>;

class PyramidView{
public:
    PyramidView(std::string_view image_path,  image_map& map):
        image_coll_path(image_path), 
        base_image_map(map)
        {}
    
    void AssembleBaseLevel();
    void GeneratePyramid(std::optional<image_map> map);


private:
    std::string image_coll_path, base_zarr_path, pyramid_zarr_path;
    std::uint16_t max_level;
    image_map base_image_map;
};
}


