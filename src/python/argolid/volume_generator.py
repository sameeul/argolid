from bfio import BioReader
import filepattern as fp
import os
import tensorstore as ts
import numpy as np
import concurrent.futures
from multiprocessing import get_context
import json
from typing import List, Tuple

CHUNK_SIZE = 1024


class VolumeGenerator:
    """
    A class for generating volumetric data from image files and storing it in a Zarr array.

    This class handles the process of reading image files, grouping them based on specified criteria,
    and writing the data into a Zarr array for efficient storage and access of large-scale volumetric data.

    Attributes:
        _source_dir (str): Directory containing the source image files.
        _group_by (str): Criterion for grouping images ('c', 't', or 'z').
        _file_pattern (str): Pattern to match the image files.
        _out_dir (str): Output directory for the generated Zarr array.
        _out_name (str): Name of the output Zarr array.
        _X (int): Width of the images.
        _Y (int): Height of the images.
        files (List[str]): List of image file paths.
        _zarr_spec (dict): Specification for the Zarr array.
    """

    def __init__(
        self,
        source_dir: str,
        group_by: str,
        file_pattern: str,
        out_dir: str,
        out_name: str,
    ) -> None:
        self._source_dir: str = source_dir
        self._group_by: str = group_by
        self._file_pattern: str = file_pattern
        self._out_dir: str = out_dir
        self._out_name: str = out_name
        self._X: int
        self._Y: int
        self.files: List[str]
        self._zarr_spec: dict

    def init_base_zarr_file(self):
        # find out what are the dimension of the zarr array
        fps = fp.FilePattern(self._source_dir, self._file_pattern)
        groups = [fi[0] for fi, _ in fps(group_by=self._group_by)]
        dimensions = [v for t in groups for v in t if isinstance(v, int)]
        dim_min = min(dimensions)
        dim_max = max(dimensions)
        replace_value = f"({dim_min}-{dim_max})"
        # Get the number of layers to stack
        dim_size = len(dimensions)
        self.files = []
        for file in fps():
            self.files.append(file[1][0])
        # open a test file
        with BioReader(self.files[0], backend="tensorstore") as br:
            self._X = br.X
            self._Y = br.Y
            dtype = br.dtype

        self._Z = 1
        self._T = 1
        self._C = 1

        if self._group_by == "c":
            self._C = dim_size
        elif self._group_by == "t":
            self._T = dim_size
        elif self._group_by == "z":
            self._Z = dim_size
        else:
            pass

        self._zarr_spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": f"{self._out_dir}/{self._out_name}/0",
            },
            "create": True,
            "delete_existing": False,
            "open": True,
            "metadata": {
                "shape": [self._C, self._Z, self._Y, self._X],
                "chunks": [1, 1, CHUNK_SIZE, CHUNK_SIZE],
                "dtype": np.dtype(dtype).str,
                "dimension_separator": "/",
                "compressor": {
                    "id": "blosc",
                    "cname": "zstd",
                    "clevel": 1,
                    "shuffle": 1,
                    "blocksize": 0,
                },
            },
            "context": {
                "cache_pool": {},
                "data_copy_concurrency": {"limit": os.cpu_count() // 2},
                "file_io_concurrency": {"limit": os.cpu_count() // 2},
                "file_io_sync": False,
            },
        }

    def layer_writer(self, args: Tuple[str, dict, int, int]) -> None:
        """
        Write a single layer of the volume to the Zarr array.

        Args:
            args (Tuple[str, dict, int, int]): Tuple containing input file path, Zarr specification, z-index, and c-index.
        """
        input_file: str = args[0]
        zarr_spec: dict = args[1]
        z: int = args[2]
        c: int = args[3]

        zarr_array: ts.TensorStore = ts.open(zarr_spec).result()
        write_futures: List[ts.Future[None]] = []
        try:
            br: BioReader = BioReader(input_file, backend="tensorstore")
            for y in range(0, br.Y, CHUNK_SIZE):
                y_max: int = min([br.Y, y + CHUNK_SIZE])
                for x in range(0, br.X, CHUNK_SIZE):
                    x_max: int = min([br.X, x + CHUNK_SIZE])
                    write_futures.append(
                        zarr_array[c, z, y:y_max, x:x_max].write(
                            br[y:y_max, x:x_max, 0, 0, 0].squeeze()
                        )
                    )

            for future in write_futures:
                future.result()
        except Exception as e:
            print(f"Caught an exception for item : {e}")

    def generate_volume(self):
        """
        Generate the complete volume by initializing the Zarr file and writing the image stack.
        """
        self.init_base_zarr_file()
        self.write_image_stack()

    def write_image_stack(self):
        """
        Write the entire image stack to the Zarr array using parallel processing.
        """
        count = 0
        arg_list = []
        for file_name in self.files:
            t = 0
            c = 0
            z = 0
            if self._group_by == "c":
                c = count
            elif self._group_by == "t":
                c = count
            elif self._group_by == "z":
                z = count

            arg_list.append((file_name, self._zarr_spec, z, c, t))
            count += 1
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=os.cpu_count() // 2, mp_context=get_context("spawn")
        ) as executor:
            executor.map(self.layer_writer, arg_list)


    def generate_downsampled_images(self):
        min_dim = min(self._X, self._Y, self._Z)
        max_level = int(np.log2(min_dim))
        min_level = int(np.log2(512))
        base_level = 0
        pyr_gen = PyramidGenerator3D(f"{self._out_dir}/{self._out_name}",str(base_level))
        for level in range(base_level, base_level+max_level-min_level):
            self.downsample_pyramid(level)



class PyramidGenerator3D:
    def __init__(self, zarr_loc_dir, base_level):
        self._zarr_loc_dir = zarr_loc_dir
        self._base_level = base_level

    def downsample_pyramid(self, level):

        ds_spec = {
            "driver": "downsample",
            "downsample_factors": [1, 2**level, 2**level, 2**level],
            "downsample_method": "mean",
            "base": {
                "driver": "zarr",
                "kvstore": {
                    "driver": "file",
                    "path": f"{self._zarr_loc_dir}/{self._base_level}",
                },
            },
        }

        ds_zarr_array = ts.open(ds_spec).result()
        [C, Z, Y, X] = ds_zarr_array.shape
        ds_write_spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": f"{self._zarr_loc_dir}/{level}",
            },
            "create": True,
            "delete_existing": True,
            "metadata": {
                "shape": ds_zarr_array.shape,
                "chunks": [1, 1, CHUNK_SIZE, CHUNK_SIZE],
                "dtype": np.dtype(ds_zarr_array.dtype.numpy_dtype).str,
                "dimension_separator": "/",
                "compressor": {
                    "id": "blosc",
                    "cname": "zstd",
                    "clevel": 1,
                    "shuffle": 1,
                    "blocksize": 0,
                },
            },
        }

        ds_zarr_array_write = ts.open(ds_write_spec).result()
        write_futures = []
        for c in range(C):
            for z in range(Z):
                for y in range(0, Y, CHUNK_SIZE):
                    y_max = min([Y, y + CHUNK_SIZE])
                    for x in range(0, X, CHUNK_SIZE):
                        x_max = min([X, x + CHUNK_SIZE])
                        write_futures.append(
                            ds_zarr_array_write[c, z, y:y_max, x:x_max].write(
                                ds_zarr_array[c, z, y:y_max, x:x_max].read().result()
                            )
                        )

        for future in write_futures:
            future.result()

    def _create_zattr_file(self, min_level, num_levels) -> None:
        """
        Creates a .zattrs file for the zarr pyramid.
        """
        attr_dict: dict = {}

        axes_metadata = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        attr_dict["axes"] = axes_metadata
        multiscale_metadata: list = []
        for key in range(min_level, min_level + num_levels):
            metadata = {
                "coordinateTransformations": [
                    {"scale": [1, 2**key, 2**key, 2**key], "type": "scale"}
                ],
                "path": f"{key}",
            }
            multiscale_metadata.append(metadata)

        attr_dict["datasets"] = multiscale_metadata
        attr_dict["version"] = "0.4"
        attr_dict["name"] = "test_image"
        attr_dict["metadata"] = {"method": "mean"}

        final_attr_dict = {"multiscales": [attr_dict]}

        with open(f"{self._zarr_loc_dir}/.zattrs", "w") as json_file:
            json.dump(final_attr_dict, json_file)

    def generate_pyramid(self, start_level, end_level):

        num_levels = end_level-start_level+1
        self._create_zattr_file(self._base_level, num_levels)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=os.cpu_count() // 2, mp_context=get_context("spawn")
        ) as executor:
            executor.map(self.downsample_pyramid, range(num_levels))
