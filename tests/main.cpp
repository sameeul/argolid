#include <iostream>
#include <string>
#include "../src/ome_tiff_to_zarr_converter.h"
#include "../src/zarr_pyramid_assembler.h"
#include "../src/zarr_base_to_pyr_gen.h"
#include "../src/ome_tiff_to_zarr_pyramid.h"
#include "../src/utilities.h"
#include "BS_thread_pool.hpp"
#include<chrono>


#include "tensorstore/tensorstore.h"
#include "tensorstore/context.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/zarr/dtype.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/open.h"


void test_zarr_pyramid_writer(){
    std::string input_file = "/home/samee/Downloads/WT_Uninfected_top_right.tif";
    std::string output_file = "/home/samee/axle/data/bia_test";
    
    //input_file = "/home/samee/axle/data/test_image.ome.tif";
    //output_file = "/home/samee/axle/data/test_image_ome_zarr_2";

    auto zpw = OmeTiffToZarrConverter();
    auto t1 = std::chrono::high_resolution_clock::now();
    BS::thread_pool th_pool;
    zpw.Convert(input_file, output_file, "16", VisType::Viv ,th_pool);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> et1 = t2-t1;
    std::cout << "time for base image: "<< et1.count() << std::endl;
}

void test_zarr_pyramid_assembler()
{
    std::string input_dir = "/home/samee/axle/data/intensity1";
    std::string output_file = "/home/samee/axle/data/test_assembly";
    std::string stitch_vector = "/home/samee/axle/data/tmp_dir/img-global-positions-1.txt";

    auto zpw = OmeTiffCollToChunked();
    auto t1 = std::chrono::high_resolution_clock::now();
    BS::thread_pool th_pool;
    zpw.Assemble(input_dir, stitch_vector, output_file, "17", VisType::PCNG, th_pool);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> et1 = t2-t1;
    std::cout << "time for base image: "<< et1.count() << std::endl;

}

void test_zarr_pyramid_gen(){
    std::string input_zarr_dir = "/home/samee/axle/data/test_assembly";
    std::string output_root_dir = "/home/samee/axle/data/test_assembly_out";
    auto t1 = std::chrono::high_resolution_clock::now();
    auto zarr_pyr_gen = ChunkedBaseToPyramid();
    BS::thread_pool th_pool;
    auto channel_ds_config = std::unordered_map<std::int64_t, DSType>();
    zarr_pyr_gen.CreatePyramidImages(input_zarr_dir, output_root_dir, 17, 1024, VisType::Viv, channel_ds_config, th_pool);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> et1 = t2-t1;
    std::cout << "time for base image: "<< et1.count() << std::endl;
}

void test_ome_tiff_to_zarr_pyramid_gen(){
    std::string input_tiff_file = "/home/samee/Downloads/WT_Uninfected_top_right.tif";
    //input_tiff_file = "/home/samee/axle/data/test_image.ome.tif";
    std::string output_dir = "/mnt/hdd8/axle/data/test_assembly_out";
    auto t1 = std::chrono::high_resolution_clock::now();
    auto zarr_pyr_gen = OmeTiffToChunkedPyramid();
    auto channel_ds_config = std::unordered_map<std::int64_t, DSType>();
    zarr_pyr_gen.GenerateFromSingleFile(input_tiff_file, output_dir, 256, VisType::Viv, channel_ds_config);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> et1 = t2-t1;
    std::cout << "time for base image: "<< et1.count() << std::endl;
}

void test_ome_tiff_coll_to_zarr_pyramid_gen(){
    std::string input_dir = "/home/samee/axle/data/images";
    std::string stitch_vector = "x{x:d+}_y{y:d+}_c{c:d}.ome.tiff";
    std::string image_name = "test_image";
    std::string output_dir = "/home/samee/axle/data";
    auto t1 = std::chrono::high_resolution_clock::now();
    auto zarr_pyr_gen = OmeTiffToChunkedPyramid();
    auto channel_ds_config = std::unordered_map<std::int64_t, DSType>();
    channel_ds_config[1] = DSType::Mean;
    channel_ds_config[0] = DSType::Mode_Max;
    zarr_pyr_gen.GenerateFromCollection(input_dir, stitch_vector, image_name, output_dir, 1024, VisType::Viv, channel_ds_config);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> et1 = t2-t1;
    std::cout << "time for base image: "<< et1.count() << std::endl;
}

void test_ome_tiff_coll_to_zarr_pyramid_gen_xml(){
    std::string input_dir = "/home/samee/axle/data/intensity1";
    std::string output_file = "/home/samee/axle/data/test_assembly/test.xml";
    std::string stitch_vector = "/home/samee/axle/data/tmp_dir/img-global-positions-1.txt";
    std::string image_name = "test_image";
    auto zpw = OmeTiffCollToChunked();
    auto t1 = std::chrono::high_resolution_clock::now();
    BS::thread_pool th_pool;
    //zpw.Assemble(input_dir, stitch_vector,output_file, VisType::Viv, th_pool);
    zpw.GenerateOmeXML(image_name, output_file);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> et1 = t2-t1;
    std::cout << "time for base image: "<< et1.count() << std::endl;
}


void test_npc_write(){
    // TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, tensorstore::Open(
    //                         {{"driver", "neuroglancer_precomputed"},
    //                         {"kvstore", {
    //                             {"driver", "file"},
    //                             {"path", "/home/samee/axle/data/test_ngs"}
    //                         }},
    //                         {"multiscale_metadata", {
    //                                       {"data_type", "uint16"},
    //                                       {"num_channels", 1},
    //                                       {"type", "image"},
    //                         }},
    //                         {"scale_metadata", {
    //                                       {"encoding", "raw"},
    //                                       {"key", "1"},
    //                                       {"size", {16,16,1}},
    //                                       {"chunk_size", {4,4,1}},
    //                         }}},
    //                         tensorstore::OpenMode::create |
    //                         tensorstore::OpenMode::delete_existing,
    //                         tensorstore::ReadWriteMode::write).result());  

    // std::vector<uint16_t> test_input(6);
    // for(int i=0; i<test_input.size(); i++){
    //     test_input[i] = i+1;
    // }
    // std::cout<<"printing input\n";
    // for(auto& x: test_input){
    //     std::cout<<x<<" ";
    // }
    // std::cout<<std::endl;

    // auto array = tensorstore::Array(test_input.data(), {3, 2}, tensorstore::c_order);
    // //auto array = tensorstore::MakeArray<std::uint16_t>({{1, 2, 3}, {4, 5, 6}});

    // tensorstore::Write(tensorstore::UnownedToShared(array), store |tensorstore::Dims(2, 3).IndexSlice({0, 0})|
    //     tensorstore::Dims(0).ClosedInterval(0,2) |
    //     tensorstore::Dims(1).ClosedInterval(0,1)).value();  

    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto outstore, tensorstore::Open(
                            {{"driver", "neuroglancer_precomputed"},
                            {"kvstore", {
                                {"driver", "file"},
                                {"path", "/home/samee/axle/data/test_ngs"}
                            }}},
                            tensorstore::OpenMode::open,
                            tensorstore::ReadWriteMode::read).result()); 
    std::vector<uint16_t> test_output(6);

    auto array2 = tensorstore::Array(test_output.data(), {3, 2}, tensorstore::fortran_order);
    tensorstore::Read(outstore |tensorstore::Dims(2, 3).IndexSlice({0, 0})|
        tensorstore::Dims(0).ClosedInterval(0,2) |
        tensorstore::Dims(1).ClosedInterval(0,1), tensorstore::UnownedToShared(array2)).value();  
    
    std::cout<<"printing output\n";
    for(auto& x: test_output){
        std::cout<<x<<" ";
    }
    std::cout<<std::endl;

    // tensorstore::Write(array, store |
    //     tensorstore::Dims(0).ClosedInterval(0,1) |
    //     tensorstore::Dims(1).ClosedInterval(0,1) |
    //     tensorstore::Dims(2).ClosedInterval(0,0) |
    //     tensorstore::Dims(3).ClosedInterval(0,0)).value();  
}

void test_zarr_write(){
    TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, tensorstore::Open(
                            {{"driver", "zarr"},
                            {"kvstore", {
                                {"driver", "file"},
                                {"path", "/home/samee/axle/data/test_zarr"}
                            }},
                            {"metadata", {
                                          {"zarr_format", 2},
                                          {"shape", {1,1,16,16}},
                                          {"chunks", {1,1,4,4}},
                                          {"dtype", "<u2"},
                                          },
                            }},
                            tensorstore::OpenMode::create |
                            tensorstore::OpenMode::delete_existing,
                            tensorstore::ReadWriteMode::write).result());  

    auto array = tensorstore::MakeArray<std::uint16_t>({{1, 2}, {4, 5}});
    // tensorstore::Write(array, store |tensorstore::Dims(2, 3).IndexSlice({0, 0})|
    //     tensorstore::Dims(0).ClosedInterval(0,1) |
    //     tensorstore::Dims(1).ClosedInterval(0,1)).value();  

    tensorstore::Write(array, store |
        tensorstore::Dims(2).ClosedInterval(0,1) |
        tensorstore::Dims(3).ClosedInterval(0,1) |
        tensorstore::Dims(0).ClosedInterval(0,0) |
        tensorstore::Dims(1).ClosedInterval(0,0)).value();  
}


int main(){
    std::cout<<"hello"<<std::endl;
    //test_zarr_pyramid_writer();
    //test_zarr_pyramid_assembler();
    //test_zarr_pyramid_gen();
    //test_ome_tiff_to_zarr_pyramid_gen();
    test_ome_tiff_coll_to_zarr_pyramid_gen();
    //test_ome_tiff_coll_to_zarr_pyramid_gen_xml();
    //test_npc_write();
    //test_zarr_write();
}
