#include <iomanip>
#include <ctime>
#include <chrono>
#include "utilities.h"

tensorstore::Spec GetOmeTiffSpecToRead(const std::string& filename){
    return tensorstore::Spec::FromJson({{"driver", "ometiff"},

                            {"kvstore", {{"driver", "tiled_tiff"},
                                         {"path", filename}}
                            },
                            {"context", {
                              {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                              {"data_copy_concurrency", {{"limit", 8}}},
                              {"file_io_concurrency", {{"limit", 8}}},
                            }},
                            }).value();
}

tensorstore::Spec GetZarrSpecToWrite(   const std::string& filename, 
                                        std::vector<std::int64_t>& image_shape, 
                                        std::vector<std::int64_t>& chunk_shape,
                                        std::string& dtype){
    return tensorstore::Spec::FromJson({{"driver", "zarr"},
                            {"kvstore", {{"driver", "file"},
                                         {"path", filename}}
                            },
                            {"context", {
                              {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                              {"data_copy_concurrency", {{"limit", 8}}},
                              {"file_io_concurrency", {{"limit", 8}}},
                            }},
                            {"metadata", {
                                          {"zarr_format", 2},
                                          {"shape", image_shape},
                                          {"chunks", chunk_shape},
                                          {"dtype", dtype},
                                          },
                            }}).value();
}

tensorstore::Spec GetZarrSpecToRead(const std::string& filename, const std::string& scale_key){
    return tensorstore::Spec::FromJson({{"driver", "zarr"},
                            {"kvstore", {{"driver", "file"},
                                         {"path", filename+"/"+scale_key}}
                            }
                            }).value();
}


tensorstore::Spec GetNPCSpecToRead(const std::string& filename, const std::string& scale_key){
    return tensorstore::Spec::FromJson({{"driver", "neuroglancer_precomputed"},
                            {"kvstore", {{"driver", "file"},
                                         {"path", filename}}
                            },
                            {"scale_metadata", {
                                            {"key", scale_key},
                                            },
                            }}).value();

}

tensorstore::Spec GetNPCSpecToWrite(const std::string& filename, 
                                    const std::string& scale_key,
                                    const std::vector<std::int64_t>& image_shape, 
                                    const std::vector<std::int64_t>& chunk_shape,
                                    std::int64_t resolution,
                                    std::int64_t num_channels,
                                    std::string_view dtype, bool base_level){
    if (base_level){
      return tensorstore::Spec::FromJson({{"driver", "neuroglancer_precomputed"},
                              {"kvstore", {{"driver", "file"},
                                          {"path", filename}}
                              },
                              {"context", {
                                {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                                {"data_copy_concurrency", {{"limit", 8}}},
                                {"file_io_concurrency", {{"limit", 8}}},
                              }},
                              {"multiscale_metadata", {
                                            {"data_type", dtype},
                                            {"num_channels", num_channels},
                                            {"type", "image"},
                              }},
                              {"scale_metadata", {
                                            {"encoding", "raw"},
                                            {"key", scale_key},
                                            {"size", image_shape},
                                            {"chunk_size", chunk_shape},
                                            {"resolution", {resolution, resolution, 1}}
                                            },
                              }}).value();
    } else {
      return tensorstore::Spec::FromJson({{"driver", "neuroglancer_precomputed"},
                        {"kvstore", {{"driver", "file"},
                                      {"path", filename}}
                        },
                        {"context", {
                          {"cache_pool", {{"total_bytes_limit", 1000000000}}},
                          {"data_copy_concurrency", {{"limit", 8}}},
                          {"file_io_concurrency", {{"limit", 8}}},
                        }},
                        {"scale_metadata", {
                                      {"encoding", "raw"},
                                      {"key", scale_key},
                                      {"size", image_shape},
                                      {"chunk_size", chunk_shape},
                                      {"resolution", {resolution, resolution, 1}}
                                      },
                        }}).value();
    }

}

uint16_t GetDataTypeCode (std::string_view type_name){

  if (type_name == std::string_view{"uint8"}) {return 1;}
  else if (type_name == std::string_view{"uint16"}) {return 2;}
  else if (type_name == std::string_view{"uint32"}) {return 4;}
  else if (type_name == std::string_view{"uint64"}) {return 8;}
  else if (type_name == std::string_view{"int8"}) {return 16;}
  else if (type_name == std::string_view{"int16"}) {return 32;}
  else if (type_name == std::string_view{"int32"}) {return 64;}
  else if (type_name == std::string_view{"int64"}) {return 128;}
  else if (type_name == std::string_view{"float32"}) {return 256;}
  else if (type_name == std::string_view{"float64"}) {return 512;}
  else {return 2;}
}

std::string GetUTCString() {
    // Get the current UTC time
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);

    // Convert UTC time to a string
    const int bufferSize = 16; // Sufficient size for most date/time formats
    char buffer[bufferSize];
    std::tm timeInfo;

#if defined(_WIN32)
    // Use gmtime_s instead of gmtime to get the UTC time on Windows
    gmtime_s(&timeInfo, &time);
#else
    // On other platforms, use the standard gmtime function
    gmtime_r(&time, &timeInfo);
#endif
    // Format the time string (You can modify the format as per your requirements)
    std::strftime(buffer, bufferSize, "%Y%m%d%H%M%S", &timeInfo);

    return std::string(buffer);
}
