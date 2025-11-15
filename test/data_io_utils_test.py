import os
import pathlib
import unittest

import pandas as pd

os.environ["ENV_FILE"] = str(pathlib.Path(__file__).with_name("aws_test.env").resolve())
from config.env_loader import SETTINGS
from utils.data_io_utils import save_processed, load_processed
from utils.log_utils import get_logger
from utils.s3_utils import list_bucket_objects

LOGGER = get_logger("data_io_utils_test")


class DataIOUtilsTestCase(unittest.TestCase):
    def test_processed_bucket(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        s3_uri = save_processed(df, base_dir="feature_master", name="feature_master_test")
        LOGGER.info(f"S3 URI for saved file: {s3_uri}")
        obj_keys = list_bucket_objects(SETTINGS.PROCESSED_BUCKET, "feature_master/")
        LOGGER.info(f"Object keys: {obj_keys}")
        saved_df = load_processed(obj_keys[0])
        pd.testing.assert_frame_equal(df, saved_df)


if __name__ == '__main__':
    unittest.main()
