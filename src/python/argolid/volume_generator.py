from bfio import BioReader
import filepattern as fp
import os
import tensorstore as ts
import numpy as np
import concurrent.futures
from multiprocessing import get_context


CHUNK_SIZE = 1024



class VolumeGenerator:
    def __init__(self, source_dir, group_by, file_pattern, out_dir, out_name) -> None:
        self._source_dir = source_dir
        self._group_by = group_by
        self._file_pattern = file_pattern
        self._out_dir = out_dir
        self._out_name = out_name

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

        self._zarr_spec = {
            "driver":"zarr",
            "kvstore":{
                "driver":"file",
                "path" : f"{self._out_dir}/{self._out_name}/0",
            },
            "create": True,
            "delete_existing": False,
            "open": True,
            "metadata" : {
                "shape" : [t_size, c_size, z_size, self._Y, self._X],
                "chunks" : [1,1,1,CHUNK_SIZE,CHUNK_SIZE],
                "dtype" : np.dtype(dtype).str,
                "dimension_separator" : "/",
                "compressor" : {"id": "blosc", "cname": "zstd", "clevel": 1, "shuffle": 1, "blocksize": 0},
                },
            'context': {
                'cache_pool': {},
                'data_copy_concurrency': {"limit": os.cpu_count()//2},
                'file_io_concurrency': {"limit": os.cpu_count()//2},
                'file_io_sync': False,
                },
    
            }



    def layer_writer(self, args):
        input_file = args[0] 
        zarr_spec = args[1]
        z = args[2] 
        c = args[3] 
        t = args[4]

        zarr_array = ts.open(zarr_spec).result()
        write_futures = []
        try:
            br = BioReader(input_file, backend="tensorstore")
            print(input_file)
            for y in range(0, br.Y, CHUNK_SIZE):
                y_max = min([br.Y, y + CHUNK_SIZE])
                for x in range(0, br.X, CHUNK_SIZE):
                    x_max = min([br.X, x + CHUNK_SIZE])
                    write_futures.append(zarr_array[t,c,z,y:y_max, x:x_max].write(br[y:y_max, x:x_max,0,0,0].squeeze()))

            for future in write_futures:
                    future.result()
        except Exception as e:
            print(f"Caught an exception for item : {e}")


    def generate_volume(self):
        self.init_base_zarr_file()
        self.write_image_stack()

    def write_image_stack(self):
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
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()//2, mp_context=get_context("spawn")) as executor:
            executor.map(self.layer_writer, arg_list) 

class PyramidGenerator3D:
    def __init__(self, zarr_loc_dir, base_level):
        self._zarr_loc_dir = zarr_loc_dir
        self._base_level = base_level



    def downsample_pyramid(self, level):

        ds_spec = {
                    "driver": "downsample",
                    "downsample_factors": [1,1,2**level, 2**level, 2**level],
                    "downsample_method": "mean",
                    "base": {
                                "driver": "zarr", 
                                "kvstore": {            
                                    "driver":"file",
                                    "path" : f"{self._zarr_loc_dir}/{self._base_level}"
                                }
                            }
                }

        ds_zarr_array = ts.open(ds_spec).result()
        [T, C, Z, Y, X] = ds_zarr_array.shape
        ds_write_spec = {
            "driver":"zarr",
            "kvstore":{
                "driver":"file",
                "path" : f"{self._zarr_loc_dir}/{level}",
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
        write_futures = []
        for t in range(T):
            for c in range(C):
                for z in range(Z):
                    for y in range(0, Y, CHUNK_SIZE):
                        y_max = min([Y, y + CHUNK_SIZE])
                        for x in range(0, X, CHUNK_SIZE):
                            x_max = min([X, x + CHUNK_SIZE])
                            write_futures.append(ds_zarr_array_write[ t, c, z, y:y_max, x:x_max].write(ds_zarr_array[ t, c, z, y:y_max, x:x_max].read().result()))

        for future in write_futures:
            future.result()

    def generate_pyramid(self):
        num_levels = 6 # for now
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()//2, mp_context=get_context("spawn")) as executor:
            executor.map(self.downsample_pyramid, range(num_levels)) 