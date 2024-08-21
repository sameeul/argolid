from bfio import BioReader
import filepattern as fp
import re
import tensorstore as ts
import json
import numpy as np
import asyncio

CHUNK_SIZE = 1024

class PyramidGenerator3D:
    def __init__(self, source_dir, group_by, file_pattern, out_dir, out_name) -> None:
        self._source_dir = source_dir
        self._group_by = group_by
        self._file_pattern = file_pattern
        self._out_dir = out_dir
        self._out_name = out_name


    def init_base_zarr_file(self):
        # find out what are the dimension of the zarr array
        self._fps = fp.FilePattern(self._source_dir, self._file_pattern)
        groups = [fi[0] for fi, _ in self._fps(group_by=self._group_by)]
        dimensions = [v for t in groups for v in t if isinstance(v, int)]
        dim_min = min(dimensions)
        dim_max = max(dimensions)
        replace_value = f"({dim_min}-{dim_max})"
        # Get the number of layers to stack
        dim_size = len(dimensions)
        # open a test file
        for file in self._fps():
            file_name = file[1][0]
            with BioReader(file_name) as br:
                self._X = br.X
                self._Y = br.Y
                dtype = br.dtype
            break

        z_size = 1
        t_size = 1
        c_size = 1

        if self._group_by == "c":
            c_size = dim_size
        elif self._group_by == "t":
            t_size = dim_size
        elif self._group_by == "z":
            z_size = dim_size
        else:
            pass

        spec = {
            "driver":"zarr",
            "kvstore":{
                "driver":"file",
                "path" : f"{self._out_dir}/{self._out_name}/0",
            },
            "create": True,
            "delete_existing": True,
            "metadata" : {
                "shape" : [t_size, c_size, z_size, self._Y, self._X],
                "chunks" : [1,1,1,CHUNK_SIZE,CHUNK_SIZE],
                "dtype" : np.dtype(dtype).str,
                "dimension_separator" : "/",
                "compressor" : {"id": "blosc", "cname": "zstd", "clevel": 1, "shuffle": 1, "blocksize": 0},
                }       
            }

        self._zarr_array = ts.open(spec).result()

    async def write_image_stack(self):
        write_futures = []
        count = 0
        for file in self._fps():
            file_name = file[1][0]
            with BioReader(file_name) as br:
                if self._group_by == "c":
                    write_futures.append(asyncio.ensure_future(self._zarr_array[0,count:count+1,0,0:self._Y, 0:self._X].write(br[:].squeeze()))) 
                elif self._group_by == "t":
                    write_futures.append(asyncio.ensure_future(self._zarr_array[count:count+1,0,0,0:self._Y, 0:self._X].write(br[:].squeeze())))
                elif self._group_by == "z":
                    write_futures.append(asyncio.ensure_future(self._zarr_array[0,0,count:count+1,0:self._Y, 0:self._X].write(br[:].squeeze())))
                else:
                    pass
            count += 1
        await asyncio.wait(write_futures)
        # for future in write_futures:
        #     future.result()

    def write_image_stack_2(self):
        write_futures = []
        count = 0
        for file in self._fps():
            file_name = file[1][0]
            with BioReader(file_name) as br:
                if self._group_by == "c":
                    write_futures.append(self._zarr_array[0,count:count+1,0,0:self._Y, 0:self._X].write(br[:].squeeze())) 
                elif self._group_by == "t":
                    write_futures.append(self._zarr_array[count:count+1,0,0,0:self._Y, 0:self._X].write(br[:].squeeze()))
                elif self._group_by == "z":
                    write_futures.append(self._zarr_array[0,0,count:count+1,0:self._Y, 0:self._X].write(br[:].squeeze()))
                else:
                    pass
            count += 1

        for future in write_futures:
            future.result()

    async def copy_chunk(self, source_store, target_store, t, c, z, y, y_max, x, x_max):

        data = await source_store[ t, c, z, y:y_max, x:x_max].read()
        await target_store[ t, c, z, y:y_max, x:x_max].write(data)

    async def downsample_pyramid(self, factor):

        ds_spec = {
                    "driver": "downsample",
                    "downsample_factors": [1,1,factor, factor, factor],
                    "downsample_method": "mean",
                    "base": {
                                "driver": "zarr", 
                                "kvstore": {            
                                    "driver":"file",
                                    "path" : f"{self._out_dir}/{self._out_name}/0"
                                }
                            }
                }

        ds_zarr_array = ts.open(ds_spec).result()
        [T, C, Z, Y, X] = ds_zarr_array.shape
        ds_write_spec = {
            "driver":"zarr",
            "kvstore":{
                "driver":"file",
                "path" : f"{self._out_dir}/{self._out_name}/1",
            },
            "create": True,
            "delete_existing": True,
            "metadata" : {
                "shape" : ds_zarr_array.shape,
                "chunks" : [1,1,1,CHUNK_SIZE,CHUNK_SIZE],
                "dtype" : np.dtype(ds_zarr_array.dtype.numpy_dtype).str,
                "dimension_separator" : "/",
                "compressor" : {"id": "blosc", "cname": "zstd", "clevel": 1, "shuffle": 1, "blocksize": 0},
                }       
        }

        ds_zarr_array_write = ts.open(ds_write_spec).result()
        for t in range(T):
            for c in range(C):
                for z in range(Z):
                    for y in range(0, Y, CHUNK_SIZE):
                        y_max = min([Y, y + CHUNK_SIZE])
                        for x in range(0, X, CHUNK_SIZE):
                            x_max = min([X, x + CHUNK_SIZE])
                            await self.copy_chunk(ds_zarr_array, ds_zarr_array_write, t, c, z, y, y_max, x, x_max)