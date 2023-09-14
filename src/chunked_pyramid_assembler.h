#pragma once

#include <string>
#include "utilities.h"
#include "BS_thread_pool.hpp"

struct ImageSegment{
  std::string file_name;
  std::int64_t _x_grid, _y_grid, _c_grid;

  ImageSegment(std::string fname, std::int64_t x, std::int64_t y, std::int64_t c):
    file_name(fname), _x_grid(x), _y_grid(y), _c_grid(c) {}
 
  ImageSegment():
    file_name(""), _x_grid(0), _y_grid(0), _c_grid(0) {};

};

struct ImageInfo
{
  std::int64_t _full_image_height, _full_image_width, _chunk_size_x, _chunk_size_y, _num_channels;
  std::string _data_type;
};

class OmeTiffCollToChunked{

public:

    OmeTiffCollToChunked() = default;

    ImageInfo Assemble(const std::string& input_dir,
                  const std::string& pattern,
                  const std::string& output_file, 
                  const std::string& scale_key, 
                  VisType v, 
                  BS::thread_pool& th_pool);
    void GenerateOmeXML(const std::string& image_name, const std::string& output_file, ImageInfo& whole_image);
};

