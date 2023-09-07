from omepyrgen import PyramidGenerartor
from time import time

def test_single_image():
    input_file = "/home/samee/axle/data/r001_c001_z000.ome.tif"
    output_dir = "/home/samee/axle/data/test_image_ome_zarr_2"

    start_time = time()
    pyr_gen = PyramidGenerartor()

    pyr_gen.generate_from_single_image(input_file, output_dir, 1024, "Viv", "mode_max")
    end_time = time()
    print(f"elapsed time = {end_time - start_time}")

def test_image_collection():
    input_dir = "/home/samee/axle/data/images"
    file_pattern = "x{x:d+}_y{y:d+}_c{c:d}.ome.tiff"
    print(file_pattern)
    output_dir = "/home/samee/axle/data/"
    image_name = "test_image2"
    start_time = time()
    pyr_gen = PyramidGenerartor()
    pyr_gen.set_log_level(4)
    pyr_gen.generate_from_image_collection(input_dir, file_pattern, image_name, output_dir, 1024, "Viv")
    end_time = time()
    print(f"elapsed time = {end_time - start_time}")

if __name__=="__main__":
    #test_single_image()
    test_image_collection()