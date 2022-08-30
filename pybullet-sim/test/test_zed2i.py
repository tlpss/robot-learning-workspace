import time
import unittest

import pybullet as p
import pybullet_data
from pybullet_sim.assets.path import get_asset_root_folder
from pybullet_sim.zed2i import Zed2i


class TestZed2i(unittest.TestCase):
    def setUp(self):
        asset_path = get_asset_root_folder()
        if not p.isConnected():
            p.connect(p.GUI)  # or p.DIRECT for non-graphical version
            p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
            p.setGravity(0, 0, -10)
            p.loadURDF("plane.urdf", [0, 0, -1.0])
            p.loadURDF(str(asset_path / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.01])
        self.cubeID = p.loadURDF("cube.urdf", [0, 0, 0.04], globalScaling=0.1)

    def test_camera_rendering_output_shapes(self):
        cam = Zed2i([0, -0.3001, 1], target_position=[0, -0.3, 0])
        img, depth, segm = cam.get_image()

        self.assertEqual(img.shape[0], cam.image_size[1])  # width
        self.assertEqual(img.shape[2], 3)  # RGB channels last

        self.assertEqual(depth.shape[0], cam.image_size[1])  # HxWx1
        time.sleep(2)  # allow to observe output

    def test_depth_map_contains_distance_to_xy_plane(self):
        cam = Zed2i([1, 0, 0])
        img, depth, segm = cam.get_image()
        # check if depth image is z-image in meters.
        cube_distance = depth[cam.image_size[1] // 2, cam.image_size[0] // 2]
        cube_distance2 = depth[cam.image_size[1] // 2 - 40, cam.image_size[0] // 2]

        # check this is really the z-buffer, not the distance to the eye of the camera
        # note that the color map in the GUI does not clearly show the depth differences.
        self.assertAlmostEqual(
            cube_distance, 0.95
        )  # camera at 1m of origin, cube has width of 10cm, centered on origin
        self.assertAlmostEqual(
            cube_distance2, 0.95
        )  # camera at 1m of origin, cube has width of 10cm, centered on origin
        time.sleep(2)

    def test_segmentation_ID(self):
        cam = Zed2i([1, 0, 0])
        img, depth, segm = cam.get_image()
        # check if segmentation matches the cube ID
        assert segm[cam.image_size[1] // 2, cam.image_size[0] // 2] == self.cubeID


if __name__ == "__main__":
    unittest.main()
