from .libargolid import OmeTiffToChunkedPyramidCPP, VisType, DSType, PyramidViewCPP

class PyramidGenerartor:
    def __init__(self, log_level = None) -> None:
        self._pyr_generator = OmeTiffToChunkedPyramidCPP()
        self.vis_types_dict ={ "NG_Zarr" : VisType.NG_Zarr, "PCNG" : VisType.PCNG, "Viv" : VisType.Viv}
        self.ds_types_dict = {"mean" : DSType.Mean, "mode_max" : DSType.Mode_Max, "mode_min" : DSType.Mode_Min}

    def generate_from_single_image(self, input_file, output_dir, min_dim, vis_type, ds_dict = {}):
        
        channel_ds_dict = {}
        for c, ds in ds_dict:
            channel_ds_dict[c] = self.ds_types_dict[ds]
        self._pyr_generator.GenerateFromSingleFile(input_file, output_dir, min_dim, self.vis_types_dict[vis_type], channel_ds_dict)

    def generate_from_image_collection(self, collection_path, pattern , image_name, output_dir, min_dim, vis_type, ds_dict = {}):  
        channel_ds_dict = {}
        for c in ds_dict:
            channel_ds_dict[c] = self.ds_types_dict[ds_dict[c]]
        self._pyr_generator.GenerateFromCollection(collection_path, pattern , image_name, output_dir, min_dim, self.vis_types_dict[vis_type], channel_ds_dict)

    def set_log_level(self, level):
        self._pyr_generator.SetLogLevel(level)

class PyramidView:
    def __init__(self, image_path, pyramid_zarr_loc, output_image_name, metadata_dict, log_level = None) -> None:
        x_border = (lambda d: d["x_spacing"] if "x_spacing" in d else 0)(metadata_dict)
        y_border = (lambda d: d["y_spacing"] if "y_spacing" in d else 0)(metadata_dict)
        self._pyr_view = PyramidViewCPP(image_path, pyramid_zarr_loc, output_image_name, x_border, y_border)
        self.vis_types_dict ={ "NG_Zarr" : VisType.NG_Zarr, "Viv" : VisType.Viv}
        self.ds_types_dict = {"mean" : DSType.Mean, "mode_max" : DSType.Mode_Max, "mode_min" : DSType.Mode_Min}
        if "minimum_dimension" in metadata_dict:
            self._min_dim = metadata_dict["minimum_dimension"]
        else:
            self._min_dim = 512
        if "output_type" in metadata_dict:
            self._vis_type = self.vis_types_dict[metadata_dict["output_type"]]
        else:
            self._vis_type = VisType.Viv
        self._channel_downsample_config = {}
        if "downsampling_config" in metadata_dict:
            for c in metadata_dict["downsampling_config"]:
                self._channel_downsample_config[c] = self.ds_types_dict[metadata_dict["downsampling_config"][c]]


    def generate_pyramid(self, image_map):
        self._pyr_view.GeneratePyramid(image_map, self._vis_type, self._min_dim, self._channel_downsample_config)
