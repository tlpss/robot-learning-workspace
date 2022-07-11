from typing import Tuple, Union, List

import gym
from gym.core import ObsType

from ur_sim.assets.path import get_asset_root_folder
from ur_sim.pybullet_utils import disable_debug_rendering, enable_debug_rendering
from ur_sim.ur3e import UR3e
from ur_sim.zed2i import Zed2i
import pybullet as p
import pybullet_data
import numpy as np

import logging


class UR3ePush(gym.Env):
    """
    Action space

    push_primitive: an angle and distance (continuous, angle [0-2Pi] range [0-0.2m]
    standard: a Delta on the robot position (continous, xyz :[-0.05m, 0.05m])
    """

    # Object Properties
    goal_l2_margin = 0.05
    primitive_max_push_distance = 0.15
    primitive_robot_eef_z = 0.01
    primitive_home_pose = [-0.13, -0.21, 0.30]

    object_radius = 0.05
    object_height = 0.05

    robot_flange_radius = 0.03
    robot_motion_margin = 0.02  # margin added to the start pose of the linear trajectory to avoid collisions with the object while moving down

    max_eef_delta = 0.05 # max change in 1 action per dimension

    eef_space_bounds = (-0.35,0.2,-0.48,-0.22,0.01,0.15) # (min_x,max_x,min_y,max_y,min_z,max_z)[m]

    def __init__(self, state_observation = False, push_primitive = False, real_time=False):
        self.use_state_observation = state_observation
        self.use_push_primitive = push_primitive
        self.simulate_real_time = real_time

        self.asset_path = get_asset_root_folder()
        # initialize the topdown camera
        self.camera = Zed2i([0, -1.001, 0.4],target_position=[0,-0.3,0]) # front camera (top-down -> lots of occlusions)

        self.plane_id = None
        self.robot = None
        self.table_id = None
        self.disc_id = None
        self.target_id = None
        # keep target pose, to avoid drift due to simulator instabilities.
        self.initial_eef_pose = np.array([0.1, -0.3, 0.12, 1.0, 0, 0, 0]) # default EEF pose
        self.target_position = [-0.05,-0.35,0.001]
        self.initial_object_position = [0.1,-0.33,0.06]
        #todo: action space
        #todo: observation space
        self.reset()

    def reset(self) -> Union[ObsType, tuple[ObsType, dict]]:
        if p.isConnected():
            p.resetSimulation()
        else:
            # initialize pybullet
            physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.81)

        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1) # collision shapes

        p.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=0,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0])

        disable_debug_rendering() # will do nothing if not enabled.

        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -1.0])
        self.table_id = p.loadURDF(str(self.asset_path / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.01])
        if self.use_push_primitive:
            self.initial_eef_pose = UR3ePush.primitive_home_pose
        self.robot = UR3e(eef_start_pose=self.initial_eef_pose, simulate_real_time=self.simulate_real_time)

        self.initial_object_position[:2] = self.get_random_object_position(np.array(self.target_position[:2]))
        self.disc_id = p.loadURDF(str(self.asset_path / "cylinder"/"1:2cylinder.urdf"),self.initial_object_position,globalScaling=0.1,flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.target_id = p.loadURDF(str(self.asset_path / "cylinder"/"1:2visual_cylinder.urdf"),self.target_position,globalScaling=0.1)

        p.changeVisualShape(self.target_id,-1, rgbaColor=(0.9,0.3,0.3,0.6))

        if self.simulate_real_time:
            enable_debug_rendering()

        return self.get_current_observation()

    def _get_robot_eef_position(self) -> List:
        return self.robot.get_eef_pose()[:3].tolist()
    def _get_target_position_on_plane(self) -> List:
        return self.target_position[:2]
    def _get_object_position_on_plane(self) -> List:
        return p.getBasePositionAndOrientation(self.disc_id)[0][:2]


    def get_current_observation(self):
        # cropped rgb?
        if self.use_state_observation:
            obs = self._get_robot_eef_position()
            obs.extend(self._get_object_position_on_plane())
            obs.extend(self._get_target_position_on_plane())

        else:
            rgb,depth,_ = self.camera.get_image()
            return rgb,depth

    def _reward(self):
        # get distance between disc and target position
        # reward = - distance

        object_position = np.array(self._get_object_position_on_plane())
        target_position = np.array(self._get_target_position_on_plane())

        return - np.linalg.norm(object_position - target_position)

    def _done(self):
        object_position = np.array(self._get_object_position_on_plane())
        target_position = np.array(self._get_target_position_on_plane())

        return np.linalg.norm(object_position - target_position) < UR3ePush.goal_l2_margin

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, dict]:

        # apply action
        if self.use_push_primitive:
            angle, distance = action
            distance = np.clip(distance,0.0,UR3ePush.primitive_max_push_distance).item()
            self._execute_motion_primitive(angle,distance)
        else:
            action = np.clip(action, -UR3ePush.max_eef_delta,UR3ePush.max_eef_delta)

            eef_target_position = self.robot.get_eef_pose()[0:3] + action
            eef_target_position = self._clip_target_position(eef_target_position)
            self._move_robot(eef_target_position)
        # get new observation
        new_obs = self.get_current_observation()

        # get reward
        reward = self._reward()
        # get done
        done = self._done()
        # do one additional simulation step to assure that even without robot movement the physics are updated.
        p.stepSimulation()

        return new_obs, reward, done, {}

    def render(self, mode="human"):
        return self.camera.get_image()[0]

    def _execute_motion_primitive(self, angle: float, length: float) -> bool:
        """
        Do the "motion primitive": a push along the desired angle and over the specified distance
        To avoid collisions this is executed as:
        - move to pre-start pose
        - move to start pose
        - push
        - move to post-push pose
        - move to "out-of-sight" pose (home)
        angle: radians in [0,2Pi]
        lenght: value in [0, max_pushing_distance]
        Returns True after executing if the motion was allowed
        (start robot position is in the robot workspace, end object position is in the block workspace)
        and False otherwise.
        """

        # get current position of the object
        current_object_position = np.array(self._get_object_position_on_plane())

        # determine primitive motion start and endpoint
        push_direction = np.array([np.cos(angle), np.sin(angle)])

        block_start_point = current_object_position
        robot_start_point = block_start_point - push_direction * (
            UR3ePush.object_radius + UR3ePush.robot_flange_radius + UR3ePush.robot_motion_margin
        )
        block_end_point = block_start_point + length * push_direction
        robot_end_point = block_end_point - push_direction * (
            UR3ePush.object_radius + UR3ePush.robot_flange_radius
        )

        logging.debug(f"motion primitive: (angle:{angle},len:{length} ) - {block_start_point} -> {block_end_point}")
        # calculate if the proposed primitive does not violate the robot's workspace

        if not self._position_is_in_workspace(robot_start_point):
            logging.debug(f"invalid robot startpoint for primitive {block_start_point} ->  {block_end_point}")
            return False
        if not self.position_is_in_object_space(block_end_point, margin=0.01):
            logging.debug(f"invalid  block endpoint for primitive {block_start_point} ->  {block_end_point}")
            return False

        # move to start pose
        self._move_robot(np.array([robot_start_point[0], robot_start_point[1], UR3ePush.primitive_robot_eef_z + 0.05]))
        # execute
        self._move_robot(np.array([robot_start_point[0], robot_start_point[1], UR3ePush.primitive_robot_eef_z]))
        self._move_robot(np.array([robot_end_point[0], robot_end_point[1], UR3ePush.primitive_robot_eef_z]))

        # move back to home pose
        self._move_robot(np.array([robot_end_point[0], robot_end_point[1], UR3ePush.primitive_robot_eef_z + 0.05]))
        self._move_robot(np.array([UR3ePush.primitive_home_pose[0], UR3ePush.primitive_home_pose[1], UR3ePush.primitive_home_pose[2]]))
        return True

    def _move_robot(self,position:np.array):

        eef_target_position = np.zeros(7)
        eef_target_position[4] = 1.0  # quaternion top-down eef orientation

        eef_target_position[0:3] = position
        eef_target_position = self._clip_target_position(eef_target_position)
        logging.debug(f"target EEF pose = {eef_target_position.tolist()[:3]}")

        self.robot.movep(eef_target_position)

    def calculate_optimal_primitive(self):
        current_position = self._get_object_position_on_plane()
        vector = np.array(self._get_target_position_on_plane()) - current_position
        angle = np.arctan2(vector[1], vector[0])
        length = np.linalg.norm(vector)

        # from [-pi,pi ] to [0,2pi] for normalization
        if angle < 0:
            angle += 2 * np.pi
        return angle, length

    def oracle(self):
        """
        Calculate the optimal angle and length assuming perfect observation.
        """
        angle,length = self.calculate_optimal_primitive()
        self._execute_motion_primitive(angle,length)

    @staticmethod
    def _clip_target_position(eef_target_position: np.ndarray) -> np.ndarray:
        eef_target_position[0] = np.clip(eef_target_position[0],UR3ePush.eef_space_bounds[0],UR3ePush.eef_space_bounds[1])
        eef_target_position[1] = np.clip(eef_target_position[1],UR3ePush.eef_space_bounds[2],UR3ePush.eef_space_bounds[3])
        eef_target_position[2] = np.clip(eef_target_position[2],UR3ePush.eef_space_bounds[4],UR3ePush.eef_space_bounds[5])
        return eef_target_position

    @staticmethod
    def _position_is_in_workspace(position: np.ndarray, margin: float = 0.0) -> bool:
        x, y = position[:2]

        if not (UR3ePush.eef_space_bounds[0] + margin < x < UR3ePush.eef_space_bounds[1] - margin):
            return False
        if not (UR3ePush.eef_space_bounds[2] + margin < y < UR3ePush.eef_space_bounds[3] - margin):
            return False
        return True

    @staticmethod
    def position_is_in_object_space(position: np.ndarray, margin: float = 0.0) -> bool:
        return UR3ePush._position_is_in_workspace(
            position,
            margin=UR3ePush.object_radius
            + UR3ePush.robot_flange_radius
            + UR3ePush.robot_motion_margin
            + margin,
        )
    @staticmethod
    def get_random_object_position(goal_position: np.ndarray) -> np.ndarray:
        """
        Brute-force sample positions until one is in the allowed object workspace
        """
        while True:
            x = np.random.random() - 0.5
            y = np.random.random() * 0.25 - 0.45
            position = np.array([x, y])
            logging.debug(f"proposed object reset {position}")
            if UR3ePush._position_is_in_workspace(
                position,
                margin=1.1
                * (UR3ePush.object_radius + UR3ePush.robot_flange_radius + UR3ePush.robot_motion_margin),
            ) and not np.linalg.norm(position-goal_position) < UR3ePush.goal_l2_margin:
                return position

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    env = UR3ePush(real_time=True,push_primitive=True)
    done = True
    while True:
        if done:
            obs = env.reset()
        angle,distance = env.calculate_optimal_primitive()
        obs, reward, done, _ = env.step(np.array([angle,distance]))
        print(done)
        # env.render()

