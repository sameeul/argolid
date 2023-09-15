#pragma once
#include <string>
#include <memory>
#include <vector>
#include<cmath>

#include "tensorstore/tensorstore.h"
#include "tensorstore/spec.h"

namespace argolid {
enum VisType {Viv, NG_Zarr, PCNG};

enum class DSType {Mean, Mode_Max, Mode_Min};

tensorstore::Spec GetOmeTiffSpecToRead(const std::string& filename);
tensorstore::Spec GetZarrSpecToRead(const std::string& filename, const std::string& scale_key);
tensorstore::Spec GetZarrSpecToWrite(   const std::string& filename, 
                                        const std::vector<std::int64_t>& image_shape, 
                                        const std::vector<std::int64_t>& chunk_shape,
                                        const std::string& dtype);
tensorstore::Spec GetNPCSpecToRead(const std::string& filename, const std::string& scale_key);
tensorstore::Spec GetNPCSpecToWrite(const std::string& filename, 
                                    const std::string& scale_key,
                                    const std::vector<std::int64_t>& image_shape, 
                                    const std::vector<std::int64_t>& chunk_shape,
                                    std::int64_t resolution,
                                    std::int64_t num_channels,
                                    std::string_view dtype,
                                    bool base_level);


uint16_t GetDataTypeCode (std::string_view type_name);
std::string GetUTCString();
} // ns argolid