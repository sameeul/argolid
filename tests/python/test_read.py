import unittest
import io, pathlib, shutil, logging, sys
import bfio
import numpy as np
import tensorstore as ts
import argolid
import ome_types

TEST_DIR = pathlib.Path(__file__).with_name("data")

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("bfio.test")

if "-v" in sys.argv:
    logger.setLevel(logging.INFO)

def setUpModule():
    """Create images for testing"""
    TEST_DIR.mkdir(exist_ok=True)
    for ch in range(4):
        for r in range(3):
            for c in range(2):
                image_name = f"{TEST_DIR}/test_image_r{r}_c{c}_ch{ch}.ome.tiff"
                with bfio.BioWriter(image_name, backend="python", X=1024, Y=1024, C=1, Z=1, T=1) as bw:
                    test_val = np.ones((1024,1024), dtype=np.uint16)
                    # setting up test data for mean sampling
                    if r==0 and c==0:
                        test_val[0,0] = 8
                        test_val[0,1] = 9
                        test_val[1,0] = 7
                        test_val[1,1] = 14
                    # setting up test data for mode sampling
                    if r==0 and c==1 and ch==1:
                        test_val[0,0] = 8
                        test_val[0,1] = 8
                        test_val[1,0] = 9
                        test_val[1,1] = 9

                    bw[0:1024, 0:1024, 0, 0, 0] = test_val 



class TestSingleChannelVivPyramidFromImageCollection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        input_dir = f"{TEST_DIR}"
        file_pattern = "test_image_r{x:d+}_c{y:d+}_ch0.ome.tiff"
        output_dir = f"{TEST_DIR}"
        image_name = "single_channel_image_viv"
        self._image_name = image_name + ".zarr"
        pyr_gen = argolid.PyramidGenerartor()
        pyr_gen.set_log_level(4)
        pyr_gen.generate_from_image_collection(input_dir, file_pattern, image_name, output_dir, 512, "Viv")

    def test_omexml_metadata_exists(self):
        ome_xml_path = f"{TEST_DIR}/{self._image_name}/METADATA.ome.xml"
        assert pathlib.Path(ome_xml_path).is_file() == True


    def test_valid_omexml_metadata(self):
        ome_xml_path = f"{TEST_DIR}/{self._image_name}/METADATA.ome.xml"
        ome_metadata = ome_types.from_xml(ome_xml_path)
        assert len(ome_metadata.images) == 1
        assert len(ome_metadata.images[0].pixels.channels) == 1
        assert ome_metadata.images[0].pixels.type == ome_types.model.PixelType.UINT8
        assert ome_metadata.images[0].pixels.size_x == 3072
        assert ome_metadata.images[0].pixels.size_y == 2048
        assert ome_metadata.images[0].pixels.size_z == 1
        assert ome_metadata.images[0].pixels.size_c == 1
        assert ome_metadata.images[0].pixels.size_t == 1
        assert ome_metadata.images[0].pixels.dimension_order == ome_types.model.Pixels_DimensionOrder.XYZCT


    def test_num_pyramid_levels(self):
        for i in range(5):
            assert pathlib.Path(TEST_DIR.joinpath(f"{self._image_name}/data.zarr/0/{i}")).is_dir() == True


class TestMultipleChannelVivPyramidFromImageCollection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        input_dir = f"{TEST_DIR}"
        file_pattern = "test_image_r{x:d+}_c{y:d+}_ch{c:d}.ome.tiff"
        output_dir = f"{TEST_DIR}"
        image_name = "multiple_channel_image_viv"
        self._image_name = image_name + ".zarr"
        pyr_gen = argolid.PyramidGenerartor()
        pyr_gen.set_log_level(4)
        pyr_gen.generate_from_image_collection(input_dir, file_pattern, image_name, output_dir, 1024, "Viv")

    def test_valid_omexml_metadata(self):
        ome_xml_path = f"{TEST_DIR}/{self._image_name}/METADATA.ome.xml"
        ome_metadata = ome_types.from_xml(ome_xml_path)
        assert len(ome_metadata.images) == 1
        assert len(ome_metadata.images[0].pixels.channels) == 4
        assert ome_metadata.images[0].pixels.type == ome_types.model.PixelType.UINT8
        assert ome_metadata.images[0].pixels.size_x == 3072
        assert ome_metadata.images[0].pixels.size_y == 2048
        assert ome_metadata.images[0].pixels.size_z == 1
        assert ome_metadata.images[0].pixels.size_c == 4
        assert ome_metadata.images[0].pixels.size_t == 1
        assert ome_metadata.images[0].pixels.dimension_order == ome_types.model.Pixels_DimensionOrder.XYZCT

    def test_num_pyramid_levels(self):
        for i in range(4):
            assert pathlib.Path(TEST_DIR.joinpath(f"{self._image_name}/data.zarr/0/{i}")).is_dir() == True


    def test_base_layer_data(self):
        dataset_future = ts.open({  'driver':'zarr', 
                                    'kvstore':
                                        {'driver':'file', 
                                         'path':f'{TEST_DIR}/{self._image_name}/data.zarr/0/0'
                                        }
                                })
        dataset = dataset_future.result()
        assert dataset.shape == (1, 4, 1, 2048, 3072)
# test Viv compatible Zarr is produced
# test OmeXml metadata
    # num channels
# test correct number of pyramid layers are produced
# test single channel image
# test multi channel image
# test proper downsampling is done
    # mode_max, mode_min, mean
# test channel specific downsampling

# test Neuroglance compatible Zarr is produced

# test Precomputed Neuroglancer is produced.
