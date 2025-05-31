import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import open3d as o3d


FOLDER_PATHS = [
    "192.168.26.26_2020-11-25_20-01-45_frame-1899_part_1",
    "192.168.26.26_2020-11-25_20-01-45_frame-2155_part_2",
    "192.168.26.26_2020-11-25_20-01-45_frame-2414_part_3",
    "192.168.26.26_2020-11-25_20-01-45_frame-2566_part_4"
]