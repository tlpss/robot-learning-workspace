import time
from typing import Tuple, Union

import gym
from gym.core import ActType, ObsType

from ur_sim.assets.path import get_asset_root_folder
from ur_sim.ur3e import UR3e
from ur_sim.zed2i import Zed2i
import pybullet as p
import pybullet_data

class UR3ePush(gym.Env):

    def __init__(self, state_observation = False, push_primitive = False, debug=False):

        # initialize pybullet
        physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.81)

        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=0,
            cameraPitch=-45,
            cameraTargetPosition=[0,0,0])


        self.asset_path = get_asset_root_folder()
        # initialize the topdown camera
        self.camera = Zed2i([0, -0.3001, 1],target_position=[0,-0.3,0]) # top down camera

        self.plane_id = None
        self.robot = None
        self.table_id = None
        self.disc_id = None
        self.target_id = None

    def reset(self) -> Union[ObsType, tuple[ObsType, dict]]:
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -1.0])

        self.table_id = p.loadURDF(str(self.asset_path / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.01])
        self.robot = UR3e()
        self.disc_id = p.loadURDF(str(self.asset_path / "cylinder"/"1:2cylinder.urdf"),[0.25,-0.2,0.06],globalScaling=0.1)
        self.target_id = p.loadURDF(str(self.asset_path / "cylinder"/"1:2visual_cylinder.urdf"),[0,-0.25,0.001],globalScaling=0.1)

    def _observation(self):
        # cropped rgb?
        rgb,_,_ = self.camera.get_image()
        return rgb
    def _reward(self):
        # get distance between disc and target position
        # reward = - distance
        pass
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # apply action
        # get new observation
        # get reward
        # get done
        pass

    def render(self, mode="human"):
        pass


if __name__ == "__main__":
    env = UR3ePush()
    env.reset()
    time.sleep(20)
