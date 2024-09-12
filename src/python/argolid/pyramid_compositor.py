from pathlib import Path
import json
import os
import math
import shutil

import numpy as np
import ome_types
import tensorstore as ts

CHUNK_SIZE = 1024

OME_DTYPE = {
    "uint8": ome_types.model.PixelType.UINT8,
    "int8": ome_types.model.PixelType.INT8,
    "uint16": ome_types.model.PixelType.UINT16,
    "int16": ome_types.model.PixelType.INT16,
    "uint32": ome_types.model.PixelType.UINT32,
    "int32": ome_types.model.PixelType.INT32,
    "float": ome_types.model.PixelType.FLOAT,
    "double": ome_types.model.PixelType.DOUBLE,
}


def get_zarr_read_spec(file_path):
    return {
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": file_path,
        },
        "open": True,
        "context": {
            "cache_pool": {},
            "data_copy_concurrency": {"limit": os.cpu_count()},
            "file_io_concurrency": {"limit": os.cpu_count()},
            "file_io_sync": False,
        },
    }


def get_zarr_write_spec(file_path, chunk_size, base_shape, dtype):
    return {
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": file_path,
        },
        "create": True,
        "delete_existing": False,
        "open": True,
        "metadata": {
            "shape": base_shape,
            "chunks": [1, 1, 1, chunk_size, chunk_size],
            "dtype": np.dtype(dtype).str,
            # "dimension_separator" : "/",
            # "compressor" : {"id": "blosc", "cname": "zstd", "clevel": 1, "shuffle": 1, "blocksize": 0},
        },
        "context": {
            "cache_pool": {},
            "data_copy_concurrency": {"limit": os.cpu_count()},
            "file_io_concurrency": {"limit": os.cpu_count()},
            "file_io_sync": False,
        },
    }


class PyramidCompositor:
    def __init__(self, well_pyramid_loc, out_dir, pyramid_file_name) -> None:
        self._well_pyramid_loc = well_pyramid_loc
        self._tile_cache = None
        self._pyramid_file_name = f"{out_dir}/{pyramid_file_name}"
        self._ome_metadata_file = f"{out_dir}/{pyramid_file_name}/METADATA.ome.xml"
        self._well_map = None

    def _create_xml(self) -> ome_types.model.OME:
        ome_metadata = ome_types.model.OME()
        ome_metadata.images.append(
            ome_types.model.Image(
                id="Image:0",
                pixels=ome_types.model.Pixels(
                    id="Pixels:0",
                    dimension_order="XYZCT",
                    big_endian=False,
                    size_c=self._plate_image_shapes[0][1],
                    size_z=self._plate_image_shapes[0][2],
                    size_t=self._plate_image_shapes[0][0],
                    size_x=self._plate_image_shapes[0][4],
                    size_y=self._plate_image_shapes[0][3],
                    channels=[
                        ome_types.model.Channel(
                            id=f"Channel:{i}",
                            samples_per_pixel=1,
                        )
                        for i in range(self._plate_image_shapes[0][2])
                    ],
                    type=OME_DTYPE[str(self._image_dtype)],
                ),
            )
        )

        with open(self._ome_metadata_file, "w") as fw:
            fw.write(str(ome_metadata.to_xml()))

    def _create_zattr_file(self):
        attr_dict = {}
        multiscale_metadata = []
        for key in self._plate_image_shapes:
            multiscale_metadata.append({"path": str(key)})
        attr_dict["datasets"] = multiscale_metadata
        attr_dict["version"] = "0.1"
        attr_dict["name"] = self._pyramid_file_name
        attr_dict["metadata"] = {"method": "mean"}

        final_attr_dict = {"multiscales": [attr_dict]}

        with open(f"{self._pyramid_file_name}/data.zarr/0/.zattrs", "w") as json_file:
            json.dump(final_attr_dict, json_file)

    def _create_zgroup_file(self):
        zgroup_dict = {"zarr_format": 2}

        with open(f"{self._pyramid_file_name}/data.zarr/0/.zgroup", "w") as json_file:
            json.dump(zgroup_dict, json_file)

        with open(f"{self._pyramid_file_name}/data.zarr/.zgroup", "w") as json_file:
            json.dump(zgroup_dict, json_file)

    def _create_auxilary_files(self):
        # create ome xml metadata
        self._create_xml()
        # create zattrs file
        self._create_zattr_file()
        # create zgroup file
        self._create_zgroup_file()

    def _write_tile_data(self, level, channel, y_index, x_index):
        y_range = [
            y_index * CHUNK_SIZE,
            min((y_index + 1) * CHUNK_SIZE, self._zarr_arrays[level].shape[3]),
        ]
        x_range = [
            x_index * CHUNK_SIZE,
            min((x_index + 1) * CHUNK_SIZE, self._zarr_arrays[level].shape[4]),
        ]

        assembled_width = x_range[1] - x_range[0]
        assembled_height = y_range[1] - y_range[0]
        assembled_image = np.zeros(
            (assembled_height, assembled_width), dtype=self._image_dtype
        )

        # find what well images are needed

        # are we at the begining of the well image?
        well_image_height = self._well_image_shapes[level][0]
        well_image_width = self._well_image_shapes[level][1]

        row_start_pos = y_range[0]
        while row_start_pos < y_range[1]:
            row = row_start_pos // well_image_height
            local_y_start = row_start_pos - y_range[0]
            tile_y_start = row_start_pos - row * well_image_height
            tile_y_end = (row + 1) * well_image_height - row_start_pos
            col_start_pos = x_range[0]
            while col_start_pos < x_range[1]:
                col = col_start_pos // well_image_width
                local_x_start = col_start_pos - x_range[0]
                tile_x_start = col_start_pos - col * well_image_width
                tile_x_end = (col + 1) * well_image_width - col_start_pos

                # read well zarr file
                well_file_name = self._well_map.get((col, row, channel))
                zarr_file_loc = Path(well_file_name) / "data.zarr/0/"
                zarr_array_loc = zarr_file_loc / str(level)
                zarr_file = ts.open(get_zarr_read_spec(str(zarr_array_loc))).result()

                # copy data
                assembled_image[
                    local_y_start : local_y_start + tile_y_end - tile_y_start,
                    local_x_start : local_x_start + tile_x_end - tile_x_start,
                ] = (
                    zarr_file[0, 0, 0, tile_y_start:tile_y_end, tile_x_start:tile_x_end]
                    .read()
                    .result()
                )
                col_start_pos += tile_x_end - tile_x_start  # update col index

            row_start_pos += tile_y_end - tile_y_start

        zarr_array = self._zarr_arrays[level]
        zarr_array[
            0, channel, 0, y_range[0] : y_range[1], x_range[0] : x_range[1]
        ].write(assembled_image).result()

    def set_well_map(self, well_map):
        self._well_map = well_map
        self._well_image_shapes = {}
        for coord in well_map:
            file = well_map[coord]
            zarr_file_loc = Path(file) / "data.zarr/0/"
            attr_file_loc = Path(file) / "data.zarr/0/.zattrs"
            if attr_file_loc.exists():
                with open(str(attr_file_loc), "r") as f:
                    attrs = json.load(f)
                    mutliscale_metadata = attrs["multiscales"][0]["datasets"]
                    self._pyramid_levels = len(mutliscale_metadata)
                    for dic in mutliscale_metadata:
                        res_key = dic["path"]
                        zarr_array_loc = zarr_file_loc / res_key
                        zarr_file = ts.open(
                            get_zarr_read_spec(str(zarr_array_loc))
                        ).result()
                        self._well_image_shapes[int(res_key)] = (
                            zarr_file.shape[-2],
                            zarr_file.shape[-1],
                        )
                        if res_key == "0":
                            self._image_dtype = zarr_file.dtype.numpy_dtype

            break

        num_rows = 0
        num_cols = 0
        num_channels = 0

        for coord in well_map:
            num_rows = max(num_rows, coord[1])
            num_cols = max(num_cols, coord[0])
            num_channels = max(num_channels, coord[2])

        num_cols += 1
        num_rows += 1
        num_channels += 1

        self._num_channels = num_channels

        self._plate_image_shapes = {}
        self._zarr_arrays = {}
        self._tile_cache = set()
        for l in self._well_image_shapes:
            level = int(l)
            self._plate_image_shapes[level] = (
                1,
                num_channels,
                1,
                num_rows * self._well_image_shapes[level][0],
                num_cols * self._well_image_shapes[level][1],
            )
            num_row_tiles = math.ceil(
                1.0 * num_rows * self._well_image_shapes[level][0] / CHUNK_SIZE
            )
            num_col_tiles = math.ceil(
                1.0 * num_cols * self._well_image_shapes[level][1] / CHUNK_SIZE
            )
            if num_row_tiles == 0:
                num_row_tiles = 1
            if num_col_tiles == 0:
                num_col_tiles == 1
            self._zarr_arrays[level] = ts.open(
                get_zarr_write_spec(
                    f"{self._pyramid_file_name}/data.zarr/0/{level}",
                    CHUNK_SIZE,
                    self._plate_image_shapes[level],
                    np.dtype(self._image_dtype).str,
                )
            ).result()

        self._create_auxilary_files()

    def reset_composition(self):
        shutil.rmtree(self._pyramid_file_name)
        self._well_map = None
        self._plate_image_shapes = None
        self._tile_cache = None
        self._plate_image_shapes = {}
        self._zarr_arrays = {}

    def get_tile_data(self, level, channel, y_index, x_index):

        if self._well_map is None:
            print("No well map is set. Unable to generate pyramid")
            return
        if level not in self._well_image_shapes:
            print(f"Requested level ({level}) does not exist")
            return

        if channel >= self._num_channels:
            print(f"Requested channel ({channel}) does not exist")
            return

        if y_index > (self._plate_image_shapes[level][3] // CHUNK_SIZE):
            print(f"Requested y index ({y_index}) does not exist")
            return

        if x_index > (self._plate_image_shapes[level][4] // CHUNK_SIZE):
            print(f"Requested y index ({x_index}) does not exist")
            return

        if (level, channel, y_index, x_index) in self._tile_cache:
            return
        else:
            self._write_tile_data(level, channel, y_index, x_index)
            self._tile_cache.add((level, channel, y_index, x_index))
            return
