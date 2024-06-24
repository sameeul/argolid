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
    def __init__(self, image_path, pyramid_zarr_loc, output_image_name,  image_map, vis_type, log_level = None) -> None:
        base_zarr_loc = pyramid_zarr_loc + "/base_zarr_loc"
        self._pyr_view = PyramidViewCPP(image_path, base_zarr_loc, pyramid_zarr_loc, output_image_name, image_map)
        self.vis_types_dict ={ "NG_Zarr" : VisType.NG_Zarr, "Viv" : VisType.Viv}
        self.ds_types_dict = {"mean" : DSType.Mean, "mode_max" : DSType.Mode_Max, "mode_min" : DSType.Mode_Min}
        self._pyr_view.AssembleBaseLevel(self.vis_types_dict[vis_type])

    def generate_pyramid(self, min_dim, vis_type, ds_dict = {}):
        channel_ds_dict = {}
        for c in ds_dict:
            channel_ds_dict[c] = self.ds_types_dict[ds_dict[c]]
        self._pyr_view.GeneratePyramid(None, self.vis_types_dict[vis_type], min_dim, channel_ds_dict )

    def regenerate_pyramid(self, image_map, min_dim, vis_type, ds_dict = {}):
        channel_ds_dict = {}
        for c in ds_dict:
            channel_ds_dict[c] = self.ds_types_dict[ds_dict[c]]
        self._pyr_view.GeneratePyramid(image_map, self.vis_types_dict[vis_type], min_dim, channel_ds_dict )