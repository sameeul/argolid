from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional, List
from .libargolid import OmeTiffToChunkedPyramidCPP, VisType, DSType, PyramidViewCPP

class Downsample(BaseModel):
    channel_name: str
    method: str

    @field_validator('method', mode='before')
    def check_method_config(cls, v):
        if v not in {"mean", "mode_max", "mode_min"}:
            raise ValueError(f'Value must be "mean", mode_max or "mode_min".')
        return v       

class PlateVisualizationMetadata(BaseModel):
    output_type: str
    minimum_dimension: int
    x_spacing: int
    y_spacing: int
    channel_downsample_config: Optional[List[Downsample]] = None

    @field_validator('minimum_dimension', 'x_spacing', 'y_spacing', mode='before')
    def check_non_negative(cls, v):
        if v < 0:
            raise ValueError('value must be non-negative')
        return v

    @field_validator('output_type')
    def check_output_type_config(cls, v):
        if v not in {"Viv", "NG_Zarr"}:
                raise ValueError(f'Value be "NG_Zarr" or "Viv".')
        return v

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
    def __init__(self, image_path, pyramid_zarr_loc, output_image_name, metadata_dict:PlateVisualizationMetadata, log_level = None) -> None:
        x_border = (lambda d: d.x_spacing if hasattr(d,'x_spacing') and d.x_spacing is not None else 0)(metadata_dict)
        y_border = (lambda d: d.y_spacing if hasattr(d,'y_spacing') and d.y_spacing is not None else 0)(metadata_dict)
        self._pyr_view = PyramidViewCPP(image_path, pyramid_zarr_loc, output_image_name, x_border, y_border)
        self.vis_types_dict ={ "NG_Zarr" : VisType.NG_Zarr, "Viv" : VisType.Viv}
        self.ds_types_dict = {"mean" : DSType.Mean, "mode_max" : DSType.Mode_Max, "mode_min" : DSType.Mode_Min}
        
        if hasattr(metadata_dict,'minimum_dimension') and metadata_dict.minimum_dimension is not None: 
            self._min_dim = metadata_dict.minimum_dimension
        else:
            self._min_dim = 512

        if hasattr(metadata_dict,'output_type') and metadata_dict.output_type is not None: 
            self._vis_type = self.vis_types_dict[metadata_dict.output_type]
        else:
            self._vis_type = VisType.Viv

        self._channel_downsample_config = {}
        if hasattr(metadata_dict,'channel_downsample_config') and metadata_dict.channel_downsample_config is not None: 
            for c in metadata_dict.channel_downsample_config:
                self._channel_downsample_config[c.channel_name] = self.ds_types_dict[metadata_dict.channel_downsample_config[c.method]]

    def generate_pyramid(self, image_map):
        self._pyr_view.GeneratePyramid(image_map, self._vis_type, self._min_dim, self._channel_downsample_config)
