#pragma once

#include <string>
#include "BS_thread_pool.hpp"
#include "../utilities/utilities.h"
namespace argolid {
class OmeTiffToChunkedConverter{

public:
    OmeTiffToChunkedConverter() = default;
    void Convert(   const std::string& input_file, 
                    const std::string& output_file,
                    const std::string& scale_key,  
                    const VisType v,
                    BS::thread_pool<BS::tp::none>& th_pool
                );
};
} // ns argolid
