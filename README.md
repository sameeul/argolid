# OmePyrGen

## Build Requirements

`OmePyrGen` uses `Tensorstore` for reading and writing pixel data. So `Tensorstore` build requirements are needed to be satisfied. 
For Linux, these are the requirements:
- `GCC` 10 or later
- `Clang` 8 or later
- `Python` 3.8 or later
- `CMake` 3.24 or later
- `Perl`, for building *libaom* from source (default). Must be in `PATH`. Not required if `-DTENSORSTORE_USE_SYSTEM_LIBAOM=ON` is specified.
- `NASM`, for building *libjpeg-turbo*, *libaom*, and *dav1d* from source (default). Must be in `PATH`.Not required if `-DTENSORSTORE_USE_SYSTEM_{JPEG,LIBAOM,DAV1D}=ON` is specified.
- `GNU Patch` or equivalent. Must be in `PATH`.


## Building and Installing

Here is an example of building and installing `OmePyrGen` in a Python virtual environment.
```
python -m virtualenv venv
source venv/bin/activate
pip install cmake
git clone --recurse-submodules https://github.com/sameeul/omepyrgen.git 
cd omepyrgen
python setup.py install
```

## Usage

OmePyrGen can generate Pyramids from a single image or an image collection with a stitching vector provided. It can generate three different kind of pyramids:
- Neuroglancer compatible Zarr (NG_Zarr)
- Precomputed Neuroglancer (PCNG)
- Viv compatible Zarr (Viv)

Currently, three downsampling methods (`mean`, `mode_max` and `mode_min`) are supported. A dictionary with channel id (integer) as key and downsampling method as value can be passed to specify downsampling method for specific channel. If a channel does not exist as a key in the 
dictionary, `mean` will be used as the default downsampling method

Here is an example of generating a pyramid from a single image.
```
from omepyrgen import PyramidGenerartor
input_file = "/home/samee/axle/data/test_image.ome.tif"
output_dir = "/home/samee/axle/data/test_image_ome_zarr"
min_dim = 1024
pyr_gen = PyramidGenerartor()
pyr_gen.generate_from_single_image(input_file, output_dir, min_dim, "NG_Zarr", {0:"mode_max"})
```

Here is an example of generating a pyramid from a collection of images and a stitching vector.
```
from omepyrgen import PyramidGenerartor
input_dir = "/home/samee/axle/data/intensity1"
file_pattern = "x{x:d}_y{y:d}_c{c:d}.ome.tiff"
output_dir = "/home/samee/axle/data/test_assembly_out"
image_name = "test_image"
min_dim = 1024
pyr_gen = PyramidGenerartor()
pyr_gen.generate_from_image_collection(input_dir, file_pattern, image_name, 
                                        output_dir, min_dim, "Viv", {1:"mean"})
```
