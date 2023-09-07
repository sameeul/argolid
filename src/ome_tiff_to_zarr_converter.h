#pragma once

#include <string>
#include "BS_thread_pool.hpp"
#include "utilities.h"

class OmeTiffToZarrConverter{

public:
    OmeTiffToZarrConverter(){}
    void Convert(   const std::string& input_file, 
                    const std::string& output_file,
                    const std::string& scale_key,  
                    const VisType v,
                    BS::thread_pool& th_pool
                );
};

