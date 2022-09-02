from abc import abstractmethod
import logging
from typing import List
import pybullet as p
import pybullet_data
import gym 
from pybullet_sim.assets.path import get_asset_root_folder
from pybullet_sim.hardware.ur3e import UR3e
from pybullet_sim.hardware.robotiq2F85 import Robotiq2F85
from pybullet_sim.hardware.zed2i import Zed2i
import numpy as np 
import random 

class UR3ePick(gym.Env):
    def __init__(self, use_state_observation=False, use_motion_primitive=True, simulate_realtime=True) -> None:
        self.asset_path = get_asset_root_folder()
        self.simulate_realtime = simulate_realtime
        self.use_motion_primitive = use_motion_primitive
        self.use_state_observation = use_state_observation


        super().__init__()

    def reset(self):
        # bookkeeping (should be in the "Task" as it is about the logic of the MDP)
        self.current_episode_duration = 0

        # randomization of poses
        n_objects = 1
        colors =[(1,1,1,1),(1,1,0,1),(1,0,0,1),(0,1,0,1),(0,0,1,1),(0,1,1,1),(1,0,1,1)]
        object_config_dict = [{"path": str(self.asset_path/ "cylinder" / "1:2cylinder.urdf"), "scale":(0.03,0.05)}]
        initial_eef_pose = [0.2,-0.2,0.2,1,0,0,0]


        # creation of the environment
        if p.isConnected():
            p.resetSimulation()
        else:
            # initialize pybullet
            p.connect(p.GUI)  # or p.DIRECT for non-graphical version

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -1)        
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1) # collision shapes

        p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -1.0])
        self.table_id = p.loadURDF(str(self.asset_path / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.001])
        self.basket_id = p.loadURDF("tray/tray.urdf",[0.4,-0.2,0.01],[0,0,1,0],globalScaling=0.6,useFixedBase=True)
        self.gripper = Robotiq2F85()
        self.robot = UR3e(eef_start_pose=initial_eef_pose, gripper=self.gripper, simulate_real_time=self.simulate_realtime)

        self.object_ids = []
        for i in range(n_objects):
            object_type = random.choice(object_config_dict)
            scale = np.random.uniform(object_type["scale"][0],object_type["scale"][1])
            position = [0,0,0.1]
            position[0] = np.random.uniform(-0.2,0.15)
            position[1] = np.random.uniform(-0.45,-0.2)
            id =p.loadURDF(object_type["path"],position, globalScaling=scale)
            p.changeVisualShape(id, -1, rgbaColor=random.choice(colors))
            p.changeDynamics(id,-1,lateralFriction=5.0)
            
            for _ in range(200):
                p.stepSimulation()
            self.object_ids.append(id)

    def step(self, action: np.ndarray):
        if self.use_motion_primitive:
            self.execute_pick_primitive(action)



    def execute_pick_primitive(self, grasp_position: np.ndarray):
        pregrasp_position = np.copy(grasp_position)
        pregrasp_position[2] += 0.12

        self.gripper.open_gripper()
        self._move_robot(pregrasp_position)
        self._move_robot(grasp_position)
        self.gripper.close_gripper(max_force=100)
        self._move_robot(pregrasp_position)

        # check if object was grasped

        # if grasped, set flag
        # move to bin

    def _move_robot(self, position: np.array, speed=0.001, max_steps=1000):

        eef_target_position = np.zeros(7)
        eef_target_position[3] = 1.0  # quaternion top-down eef orientation

        eef_target_position[0:3] = position
        #eef_target_position = self._clip_target_position(eef_target_position)

        logging.debug(f"target EEF pose = {eef_target_position.tolist()[:3]}")

        self.robot.movep(eef_target_position, speed=speed, max_steps=max_steps)

    def get_oracle_action(self):
        if self.use_motion_primitive:
            return self._oracle_get_pick_position()
        
    def _oracle_get_pick_position(self) -> np.ndarray:
        # get heighest object from list
        heightest_object_id = np.argmax(np.array(self._get_object_heights(self.object_ids)))
        # get position of that object
        heighest_object_position = p.getBasePositionAndOrientation(self.object_ids[heightest_object_id])[0]
        heighest_object_position = np.array(heighest_object_position)
        heighest_object_position[2] -= 0.005 # firmer grasp
        return heighest_object_position

    @staticmethod
    def _get_object_heights(object_ids: List) -> List[float]:
        heights = []
        for id in object_ids:
            state = p.getBasePositionAndOrientation(id)
            heights.append(state[0][2])
        return heights

    
        

if __name__ == "__main__":
    import time
    env = UR3ePick()
    env.reset()
    env.step(env.get_oracle_action())
    time.sleep(30)