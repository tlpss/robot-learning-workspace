import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_sim.assets.path import get_asset_root_folder


class Zed2i:
    """
    Simluated Zed2i camera in PyBullet.

    FOV and image size are configurable to allow for simulation of the real-world flow of 
        1. capturing at FUllHD (@30 Hz)
        2. center-cropping the relevant piece of the image (reducing the FOV, keeping spatial resolution)
        3. (optionally downscaling that part (reducing spatial resolution, maintaining FOV)

    Doing this in sim would incur unnecessary rendering costs.

    """
    # default values for FullHD profile taken from
    # https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view-
    z_range = (0.25, 3)
    hw_image_size = (1920,1080)
    hw_focal_length_in_pixels = 1000
    vertical_fov = np.arctan(hw_image_size[1] / 2 / hw_focal_length_in_pixels) * 2  # pinhole 
    vertical_fov_degrees = vertical_fov * 180 / np.pi

    def __init__(self, eye_position, image_size=(1920,1080), vertical_fov_degrees = 56.0, target_position=None):
        assert self.vertical_fov_degrees <= Zed2i.vertical_fov_degrees, "fov cannot exceed the HW FOV.."
        
        self.vertical_fov_degrees = vertical_fov_degrees
        self.image_size = image_size
        self.eye_position = eye_position
        self.target_position = target_position if target_position is not None else [0, 0, 0]

        self.projection_matrix, self.view_matrix = self._get_camera_matrices()

    

    def get_image(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: RGB image (H,W,3); depth image (H,w) [meters]; segmentation (H,W) [ObjectID in pybullet].
        """
        _, _, rgb, depth, segmentation = p.getCameraImage(
            width=self.image_size[0],
            height=self.image_size[1],
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            shadow=1,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        color_image_size = (self.image_size[1], self.image_size[0], 4)
        rgb = np.array(rgb, dtype=np.uint8).reshape(color_image_size)
        rgb = rgb[:, :, :3]  # remove alpha channel

        # Get depth image.
        depth_image_size = (self.image_size[1], self.image_size[0])
        znear, zfar = Zed2i.z_range
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth

        # Get segmentation image.
        segmentation = np.uint8(segmentation).reshape(depth_image_size)
        return rgb, depth, segmentation

    def _get_camera_matrices(self):
        """
        Compute Camera intrinsics and extrinsics matrices 

        """

        # view matrix == extrinsics matrix
        look_dir = np.array(self.target_position) - np.array(self.eye_position)
        # want to have up-dir vector in the plane containing the z axis and the eye position
        # so first find normal on that plane
        # then, find up dir orthongal to look dir and that normal
        # this is a little hacky, and only works for camera positions in the upper sphere of the target position
        # not for the exact zenith!
        xy_projected_look_dir = np.copy(look_dir)
        if np.isclose(look_dir[2], 0.0):
            # special case in which the xy projection is colinear with the look_dir
            # in this case up_dir = z-axis.
            up_dir = np.array([0, 0, 1])
        elif look_dir[2] > 0.0:
            raise NotImplementedError
        else:
            xy_projected_look_dir[2] = 0
            plane_normal = np.cross(look_dir, xy_projected_look_dir)
            up_dir = np.cross(plane_normal, look_dir)
        view_matrix = p.computeViewMatrix(self.eye_position, self.target_position, up_dir)

        # projection matrix ~= intrinsics matrix, projects world coords to image coords
        # with some additional thingies to avoid z-depth information loss.
        # http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
        aspect_ratio = self.image_size[0] / self.image_size[1]
        
        projection_matrix = p.computeProjectionMatrixFOV(
            self.vertical_fov_degrees, aspect_ratio, Zed2i.z_range[0], Zed2i.z_range[1]
        )

        return projection_matrix, view_matrix
def test_camera_outputs():
    asset_path = get_asset_root_folder()
    p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -10)
    p.loadURDF("plane.urdf", [0, 0, -1.0])
    p.loadURDF(str(asset_path / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.01])
    cubeId = p.loadURDF("cube.urdf", [0, 0, 0.04], globalScaling=0.1)

    cam = Zed2i([1, 0, 0.0])
    img, depth, segm = cam.get_image()
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(depth)
    # plt.show()
    # plt.imshow(segm)
    # plt.show()

    # check if depth image is z-image in meters.
    cube_distance = depth[cam.image_size[1] // 2, cam.image_size[0] // 2]
    cube_distance2 = depth[
        cam.image_size[1] // 2 - 40, cam.image_size[0] // 2
    ]  # check this is really the z-buffer, not the distance to the eye
    assert np.isclose(cube_distance, 0.95)  # camera at 1m of origin, cube has width of 10cm, centered on origin
    assert np.isclose(cube_distance2, 0.95)

    # check if segmentation matches the cube ID
    assert segm[cam.image_size[1] // 2, cam.image_size[0] // 2] == cubeId


def explore_camera_output():
    asset_path = get_asset_root_folder()
    p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -10)
    p.loadURDF("plane.urdf", [0, 0, -1.0])
    p.loadURDF(str(asset_path / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.01])
    p.loadURDF("cube.urdf", [0, 0, 0.04], globalScaling=0.1)

    cam = Zed2i([0, -0.3001, 1], image_size=(100,100),vertical_fov_degrees=30,target_position=[0, 0.1, 0])
    img, depth, segm = cam.get_image()
    plt.imshow(img)
    plt.show()
    plt.imshow(depth)
    plt.show()
    plt.imshow(segm)
    plt.show()
    time.sleep(20)


if __name__ == "__main__":
    # test_camera_outputs()
    explore_camera_output()
