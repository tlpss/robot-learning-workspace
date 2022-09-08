from abc import abstractmethod
from dataclasses import dataclass
import dataclasses
import logging
from typing import List
import pybullet as p
import pybullet_data
import gym 
from pybullet_sim.assets.path import get_asset_root_folder
from pybullet_sim.hardware.ur3e import UR3e
from pybullet_sim.hardware.robotiq2F85 import WSG50, Robotiq2F85
from pybullet_sim.hardware.zed2i import Zed2i
import numpy as np 
import random

from pybullet_sim.pybullet_utils import disable_debug_rendering, enable_debug_rendering 

ASSET_PATH = get_asset_root_folder()

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class ObjectConfig:
    
    n_objects = 5
    colors =[(1,1,1,1),(1,1,0,1),(1,0,0,1),(0,1,0,1),(0,0,1,1),(0,1,1,1),(1,0,1,1)]
    object_list= [{"path": str(ASSET_PATH/ "cylinder" / "1:2cylinder.urdf"), "scale":(0.03,0.05)}]

class UR3ePick(gym.Env):
    image_dimensions = (256,256)

    initial_eef_pose = [0.4, 0.1,0.2,1,0,0,0] # robot should be out of view.
    pick_workspace_x_range = (-0.2,0.15)
    pick_workspace_y_range = (-0.45,-0.2)


    def __init__(self, use_spatial_action_map = False, use_motion_primitive=True, simulate_realtime=True, object_config: ObjectConfig = None) -> None:
        self.simulate_realtime = simulate_realtime
        self.use_motion_primitive = use_motion_primitive
        self.use_spatial_action_map = use_spatial_action_map

        if not self.use_motion_primitive:
            assert not self.use_spatial_action_map # can only use spatial action maps with 
            raise NotImplementedError
        if object_config is None:
            self.object_config = ObjectConfig()

        # camera on part of the workspace that is reachable and does not have the robot or bin in view
        # make camera high and very small fov, to approximate an orthographic view (Andy Zeng uses orthographic reprojection through point cloud)
        self.camera = Zed2i([-0.03, -0.3301, 1.5],vertical_fov_degrees=15, image_size=UR3ePick.image_dimensions, target_position=[-0.03, -0.33, 0])
        
    
        self.current_episode_duration = 0
        self.max_episode_duration = 2*self.object_config.n_objects


        super().__init__()

    def reset(self):
        # bookkeeping (should be in the "Task" as it is about the logic of the MDP)
        self.current_episode_duration = 0

        # creation of the environment
        if p.isConnected():
            p.resetSimulation()
        else:
            # initialize pybullet
            p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        disable_debug_rendering()  # will do nothing if not enabled.

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -2) # makes life easier..
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        
        
        p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -1.0])
        p.loadURDF(str(ASSET_PATH / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.01])

        self.table_id = p.loadURDF(str(ASSET_PATH / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.001])
        self.basket_id = p.loadURDF("tray/tray.urdf",[0.4,-0.2,0.01],[0,0,1,0],globalScaling=0.6,useFixedBase=True)
        self.gripper = WSG50(simulate_realtime=self.simulate_realtime) # DON'T USE ROBOTIQ! physics are not stable..
        self.robot = UR3e(eef_start_pose=UR3ePick.initial_eef_pose, gripper=self.gripper, simulate_real_time=self.simulate_realtime)

        self.object_ids = []
        for i in range(self.object_config.n_objects):
            object_type = random.choice(self.object_config.object_list)
            scale = np.random.uniform(object_type["scale"][0],object_type["scale"][1])
            position = [0,0,0.1]
            position[0] = np.random.uniform(UR3ePick.pick_workspace_x_range[0],UR3ePick.pick_workspace_x_range[1])
            position[1] = np.random.uniform(UR3ePick.pick_workspace_y_range[0], UR3ePick.pick_workspace_y_range[1])
            id =p.loadURDF(object_type["path"],position, globalScaling=scale)
            p.changeVisualShape(id, -1, rgbaColor=random.choice(self.object_config.colors))
            p.changeDynamics(id,-1,lateralFriction=5.0)
            self.object_ids.append(id)
            for _ in range(200):
                p.stepSimulation()
        if self.simulate_realtime:
            enable_debug_rendering()


        return self.get_current_observation()

    def step(self, action: np.ndarray):
        self.current_episode_duration += 1

        if self.use_motion_primitive:
            if self.use_spatial_action_map:
                # convert (u,v,theta) to (x,y,z,theta)
                position = self._image_coords_to_world(int(action[0]),int(action[1]),self.get_current_observation()[...,3])
                action = np.concatenate([position,np.array([action[2]])])
            self.execute_pick_primitive(action)

        # check if object was grasped
        success = self._is_grasp_succesfull()
        reward = self._reward() # before place!
        
        logger.debug(f"grasp succes = {success}")

        if success:
            # move to bin (only visually pleasing?) and 
            # remove object from list.
            self.object_ids.remove(self._get_lifted_object_id())
            self._drop_in_bin()
        # move robot back to initial pose
        self._move_robot(UR3ePick.initial_eef_pose[:3],speed=0.005)
        
        done = self._done() # after bookkeeping!
        new_observation = self.get_current_observation()
        return new_observation, reward, done, {}

    def get_current_observation(self):
        rgb, depth, _ = self.camera.get_image()
        rgb = rgb.astype(np.float32)/ 255.0 # range [0,1] as it will become float with depth map
        return np.concatenate([rgb,depth[:,:,np.newaxis]],axis=-1)

    def execute_pick_primitive(self, grasp_pose: np.ndarray):

        grasp_position = grasp_pose[:3]
        grasp_position[2] = max(grasp_position[2]-0.02,0.01) # position is top of object -> graps 2cm below unless this < 0.01cm.

        if np.linalg.norm(grasp_position) > 0.48:
            logger.info(f"grasp position was not reachable {grasp_position}")
            return 
        grasp_orientation = grasp_pose[3]
        pregrasp_position = np.copy(grasp_position)
        pregrasp_position[2] += 0.15

        self.gripper.open_gripper()
        self._move_robot(pregrasp_position,grasp_orientation,speed=0.005)
        self._move_robot(grasp_position,grasp_orientation)
        self.gripper.close_gripper(max_force=50)
        self._move_robot(pregrasp_position)


    def _reward(self) -> float:
        if self.use_motion_primitive:
            return self._is_grasp_succesfull() * 1.0

    def _done(self):
        # no attempt to verify if all objects are still reachable..
        # TODO: fix!
        done = len(self.object_ids) == 0
        done = done or self.current_episode_duration >= self.max_episode_duration
        return done 

    def _move_robot(self, position: np.array, gripper_z_orientation:float = 0.0, speed=0.001, max_steps=1000):

        eef_target_position = np.zeros(7)
        eef_target_position[3:] = p.getQuaternionFromEuler([np.pi,0.,gripper_z_orientation])
        eef_target_position[0:3] = position
        #eef_target_position = self._clip_target_position(eef_target_position)

        logger.debug(f"target EEF pose = {eef_target_position.tolist()[:3]}")

        self.robot.movep(eef_target_position, speed=speed, max_steps=max_steps)

    def get_oracle_action(self):
        if self.use_motion_primitive:
            return self._oracle_get_pick_pose()
        
    def _oracle_get_pick_pose(self) -> np.ndarray:
        # get heighest object from list
        heightest_object_id = np.argmax(np.array(self._get_object_heights(self.object_ids)))
        # get position of that object
        heighest_object_position = p.getBasePositionAndOrientation(self.object_ids[heightest_object_id])[0]
        heighest_object_position = np.array(heighest_object_position)
        heighest_object_position[2] -= 0.001 # firmer grasp
        pick_pose = np.concatenate([heighest_object_position, np.zeros((1,))])
        if not self.use_spatial_action_map:
            return pick_pose
        else:
            img_point=   np.linalg.inv(self.camera.extrinsics_matrix) @ np.concatenate([heighest_object_position, np.ones((1,))])
            coordinate = self.camera.intrinsics_matrix @ img_point[:3]
            coordinate /= coordinate[2]
            coordinate = np.clip(coordinate,0,UR3ePick.image_dimensions[0]-1) # make sure poses are reachable
            return np.concatenate([coordinate[:2],np.zeros((1,))])

    def _is_grasp_succesfull(self):
        return self.gripper.get_relative_position() < 0.95 and max(self._get_object_heights(self.object_ids)) > 0.1

    def _get_lifted_object_id(self):
        """heuristic for lifted object ID -> get the object that is heighest at that moment (assumes the gripper is lifted)
        """
        assert self._is_grasp_succesfull()
        heightest_object_index = np.argmax(np.array(self._get_object_heights(self.object_ids)))
        return self.object_ids[heightest_object_index]


    def _drop_in_bin(self):
        bin_posisition = p.getBasePositionAndOrientation(self.basket_id)[0]
        drop_position = np.array(bin_posisition) + np.array([0,0,0.15])
        self._move_robot(drop_position)
        self.gripper.open_gripper()

    def _image_coords_to_world(self,u:int, v:int, depth_map:np.ndarray) -> np.ndarray:
        img_coords = np.array([u,v,1.0])
        ray_in_camera_frame = np.linalg.inv(self.camera.intrinsics_matrix )@ img_coords
        z_in_camera_frame = depth_map[v,u] # Notice order!!
        t = z_in_camera_frame / ray_in_camera_frame[2]
        position_in_camera_frame = t* ray_in_camera_frame

        position_in_world_frame = (self.camera.extrinsics_matrix @ np.concatenate([position_in_camera_frame,np.ones((1,))]))[:3]
        return position_in_world_frame


    @staticmethod
    def _get_object_heights(object_ids: List) -> List[float]:
        heights = []
        for id in object_ids:
            state = p.getBasePositionAndOrientation(id)
            heights.append(state[0][2])
        return heights

    
        

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    #logging.basicConfig(level=logging.DEBUG)

    env = UR3ePick()
    obs = env.reset()
    done = False
    while not done:
        img = obs[:,:,:3]
        print(np.max(img))
        print(np.min(img))
        plt.imshow(obs[:,:,:3])
        plt.show()
        plt.savefig("test.jpg")
        plt.imshow(obs[:,:,-1])
        plt.show()
        u = int(input("u"))
        v= int(input("v"))
        print(obs[u,v,-1])
        position = env._image_coords_to_world(u,v,obs[:,:,-1])
        print(position)
        obs, reward, done , _ = env.step(np.concatenate([position,np.zeros((1,))]))
        #obs, reward, done ,_ = env.step(env.get_oracle_action())

    time.sleep(30)