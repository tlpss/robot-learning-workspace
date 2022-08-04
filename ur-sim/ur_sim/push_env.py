import logging
from enum import Enum
from typing import Any, List, Tuple

import gym
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image
import pickle


from ur_sim.assets.path import get_asset_root_folder
from ur_sim.demonstrations import Demonstration
from ur_sim.pybullet_utils import disable_debug_rendering, enable_debug_rendering
from ur_sim.ur3e import UR3e
from ur_sim.zed2i import Zed2i

class OracleStates(Enum):
    TO_PREPUSH = 0
    TO_PUSH = 1
    PUSH = 2


class UR3ePush(gym.Env):
    """
    Action space

    push_primitive: an angle and distance (continuous, angle [0-2Pi] range [0-0.2m]
    standard: a Delta on the robot position (continous, xyz :[-0.05m, 0.05m])

    Observation space
    #todo: complete
    state observations:
    standard:
    """

    goal_l2_margin = 0.05
    primitive_max_push_distance = 0.15
    primitive_robot_eef_z = 0.02
    primitive_home_pose = [-0.13, -0.21, 0.30]  # home position of the eef between primitive execution

    # Object Properties
    object_radius = 0.05
    object_height = 0.05

    robot_flange_radius = 0.03
    robot_motion_margin = 0.02  # margin added to the start pose of the linear trajectory to avoid collisions with the object while moving down
    max_eef_delta = 0.05  # max change in 1 action per dimension
    eef_space_bounds = (-0.35, 0.2, -0.48, -0.22, 0.015, 0.15)  # (min_x,max_x,min_y,max_y,min_z,max_z)[m]

    image_dimensions = (64, 64)  # dimensions of rgb observation

    def __init__(self, state_observation=False, push_primitive=False, real_time=False):
        super().__init__()
        self.metadata["render.modes"] = ["rgb_array"]  # required for VideoRecorder wrapper

        self.use_state_observation = state_observation
        self.use_push_primitive = push_primitive
        self.simulate_real_time = real_time

        self.asset_path = get_asset_root_folder()
        # initialize the front camera (top-down -> lots of occlusions)
        self.camera = Zed2i([0, -1.001, 0.4], target_position=[0, -0.3, 0])

        self.plane_id = None
        self.robot = None
        self.table_id = None
        self.disc_id = None
        self.target_id = None

        self.initial_eef_pose = np.array([0.1, -0.3, 0.12, 1.0, 0, 0, 0])  # default EEF pose
        self.target_position = [-0.05, -0.35, 0.001]  # default target position
        self.initial_object_position = [0.1, -0.33, 0.01]  # default

        if self.use_push_primitive:
            self.action_space = gym.spaces.Box(
                low=np.array([0.0, 0.0]), high=np.array([2 * np.pi, UR3ePush.primitive_max_push_distance])
            )
        else:
            self.action_space = gym.spaces.Box(-UR3ePush.max_eef_delta, UR3ePush.max_eef_delta, (3,))

        if self.use_state_observation:
            if self.use_push_primitive:
                self.observation_space = gym.spaces.Box(-1.0, 1.0, (4,))
            else:
                self.observation_space = gym.spaces.Box(-1.0, 1.0, (7,))
        else:
            self.observation_space = gym.spaces.Box(0, 255, UR3ePush.image_dimensions)

        # don't set this max duration too low! when it was 10, the policies got stuck in local optima,
        # as there were many invalid actions among those 10 steps the number of "real" steps was too low..
        self.max_episode_duration = 20 if self.use_push_primitive else 100
        self.current_episode_duration = 0


        self.oracle_state = OracleStates.TO_PREPUSH # FSM for the step oracle
        self.reset()

    def reset(self):
        self.current_episode_duration = 0
        self.oracle_state = OracleStates.TO_PREPUSH # reset FSM for the step oracle
        if p.isConnected():
            p.resetSimulation()
        else:
            # initialize pybullet
            p.connect(p.GUI)  # or p.DIRECT for non-graphical version

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.81)

        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1) # collision shapes

        p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

        disable_debug_rendering()  # will do nothing if not enabled.

        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -1.0])
        self.table_id = p.loadURDF(str(self.asset_path / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.001])
        if self.use_push_primitive:
            self.initial_eef_pose[:3] = UR3ePush.primitive_home_pose
        # todo: surpress pybullet output during loading of URDFs.. (see pybullet_planning repo)
        else:
            # get random positions as this improves exploration and robustness of the learned policies.
            # exploration as it will spawn close to the object every now and then, robustness as it will have to learn
            # to deal with arbitrary start positions.
            self.initial_eef_pose[:3] = self._get_random_eef_position()
        if self.robot is None:
            self.robot = UR3e(eef_start_pose=self.initial_eef_pose, simulate_real_time=self.simulate_real_time)
        else:
            self.robot.reset(self.initial_eef_pose)
        self.initial_object_position[:2] = self.get_random_object_position(np.array(self.target_position[:2]))
        self.disc_id = p.loadURDF(
            str(self.asset_path / "cylinder" / "1:2cylinder.urdf"), self.initial_object_position, globalScaling=0.1
        )
        self.target_id = p.loadURDF(
            str(self.asset_path / "cylinder" / "1:2visual_cylinder.urdf"), self.target_position, globalScaling=0.1
        )

        p.changeVisualShape(self.target_id, -1, rgbaColor=(0.9, 0.3, 0.3, 0.6))

        if self.simulate_real_time:
            enable_debug_rendering()

        for _ in range(100):
            p.stepSimulation()
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
            if self.use_push_primitive:
                obs = []
            else:
                obs = self._get_robot_eef_position()
            obs.extend(self._get_object_position_on_plane())
            obs.extend(self._get_target_position_on_plane())
            return obs
        else:
            rgb, depth, _ = self.camera.get_image()
            
            rgb = Image.fromarray(rgb)
            rgb = rgb.crop((400, 0, 1400, 1000))
            rgb = rgb.resize(UR3ePush.image_dimensions)
            rgb = np.array(rgb)

            return rgb

    def _reward(self) -> float:
        object_position = np.array(self._get_object_position_on_plane())
        target_position = np.array(self._get_target_position_on_plane())

        reward = -np.linalg.norm(object_position - target_position)
        if not self.use_push_primitive:
            # incentivize robot to move to object
            # z=0.01 to incentivize contact and hence move the object so that the other loss component is also used.
            # the scale is tricky: too low and the object distance trumps it,
            # too high and it will create a local optimum to go on top of the disc and hold still
            # 1.0 appeared to be okay. 0.3 was too low.
            reward -= 1.0 * np.linalg.norm(
                self.robot.get_eef_pose()[:3] - np.concatenate((object_position, np.array([0.01])))
            )
        reward += 10 * (
            np.linalg.norm(target_position - object_position) < UR3ePush.goal_l2_margin
        )  # greatly improves learning efficiency
        return reward

    def _done(self) -> bool:
        object_position = np.array(self._get_object_position_on_plane())
        target_position = np.array(self._get_target_position_on_plane())

        done = np.linalg.norm(target_position - object_position) < UR3ePush.goal_l2_margin
        done = done or not self.position_is_in_object_space(object_position)
        done = done or not self.current_episode_duration < self.max_episode_duration
        return done

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, dict]:
        # apply action
        if self.use_push_primitive:
            angle, distance = action
            distance = np.clip(distance, 0.0, UR3ePush.primitive_max_push_distance).item()
            self._execute_motion_primitive(angle, distance)
        else:
            action = np.clip(action, -UR3ePush.max_eef_delta, UR3ePush.max_eef_delta)

            eef_target_position = self.robot.get_eef_pose()[0:3] + action
            eef_target_position = self._clip_target_position(eef_target_position)
            if np.linalg.norm(eef_target_position) < 0.55:
                self._move_robot(eef_target_position, speed=0.002, max_steps=100)
        # get new observation
        new_obs = self.get_current_observation()

        # get reward
        reward = self._reward()
        # get done
        done = self._done()
        # do one additional simulation step to assure that even without robot movement the physics are updated.
        p.stepSimulation()

        self.current_episode_duration += 1  # must be after done calculation
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
        robot_end_point = block_end_point - push_direction * (UR3ePush.object_radius + UR3ePush.robot_flange_radius)

        logging.debug(f"motion primitive: (angle:{angle},len:{length} ) - {block_start_point} -> {block_end_point}")
        # calculate if the proposed primitive does not violate the robot's workspace

        if not self._position_is_in_workspace(robot_start_point):
            logging.debug(f"invalid robot startpoint for primitive {block_start_point} ->  {block_end_point}")
            return False
        if not self.position_is_in_object_space(block_end_point, margin=0.01):
            logging.debug(f"invalid  block endpoint for primitive {block_start_point} ->  {block_end_point}")
            return False

        # move to start pose
        self._move_robot(np.array([robot_start_point[0], robot_start_point[1], UR3ePush.primitive_robot_eef_z + 0.03]))
        # execute
        self._move_robot(
            np.array([robot_start_point[0], robot_start_point[1], UR3ePush.primitive_robot_eef_z]), speed=0.001
        )
        self._move_robot(
            np.array([robot_end_point[0], robot_end_point[1], UR3ePush.primitive_robot_eef_z]), speed=0.001
        )

        # move back to home pose
        self._move_robot(np.array([robot_end_point[0], robot_end_point[1], UR3ePush.primitive_robot_eef_z + 0.07]))
        self._move_robot(
            np.array(
                [UR3ePush.primitive_home_pose[0], UR3ePush.primitive_home_pose[1], UR3ePush.primitive_home_pose[2]]
            )
        )
        return True

    def _move_robot(self, position: np.array, speed=0.01, max_steps=1000):

        eef_target_position = np.zeros(7)
        eef_target_position[3] = 1.0  # quaternion top-down eef orientation

        eef_target_position[0:3] = position
        eef_target_position = self._clip_target_position(eef_target_position)
        logging.debug(f"target EEF pose = {eef_target_position.tolist()[:3]}")

        self.robot.movep(eef_target_position, speed=speed, max_steps=max_steps)

    def oracle_primitive_step(self):
        current_position = self._get_object_position_on_plane()
        vector = np.array(self._get_target_position_on_plane()) - current_position
        angle = np.arctan2(vector[1], vector[0])
        length = np.linalg.norm(vector)

        # from [-pi,pi ] to [0,2pi] for normalization
        if angle < 0:
            angle += 2 * np.pi
        return angle, length

    def oracle_delta_step(self):
        """
        returns an action according to the oracle policy given the action space.
        Keeps a finite state machine to check in what stage the policy is.


    d

        :return:
        """
        # hacky implementation with some code duplication, fix one day
        angle, length = self.oracle_primitive_step()
        current_object_position = np.array(self._get_object_position_on_plane())

        # determine primitive motion start and endpoint (2D as they are on the plane)
        push_direction = np.array([np.cos(angle), np.sin(angle)])

        block_start_point = current_object_position
        robot_start_point = block_start_point - push_direction * (
            UR3ePush.object_radius + UR3ePush.robot_flange_radius + UR3ePush.robot_motion_margin
        )
        block_end_point = block_start_point + length * push_direction
        robot_end_point = block_end_point - push_direction * (UR3ePush.object_radius + UR3ePush.robot_flange_radius)

        # create 3D positions from the 2D points
        robot_start_point = np.concatenate([robot_start_point,np.array([self.primitive_robot_eef_z])])
        robot_end_point = np.concatenate([robot_end_point,np.array([self.primitive_robot_eef_z])])
        pre_push_point = np.copy(robot_start_point)
        pre_push_point[2] += 0.1 # avoid collisions during approach

        current_robot_position = self.robot.get_eef_pose()[:3]
        #todo: find the prepush, push, pushtarget poses
        # keep track of which phase
        # find direction of movement and clip to the allowed box.
        # return the delta eef position

        logging.debug(f"oracle FSM state = {self.oracle_state}")
        if self.oracle_state == OracleStates.TO_PREPUSH:
            move_direction = pre_push_point - current_robot_position

        elif self.oracle_state == OracleStates.TO_PUSH:
            move_direction = robot_start_point - current_robot_position
        else:
            move_direction = robot_end_point - current_robot_position

        if np.linalg.norm(move_direction) < 0.02:
            if self.oracle_state == OracleStates.TO_PREPUSH:
                self.oracle_state = OracleStates.TO_PUSH
            elif self.oracle_state == OracleStates.TO_PUSH:
                self.oracle_state = OracleStates.PUSH

            return np.zeros(3) # little sloppy but easier.

        if np.linalg.norm(move_direction, np.inf) > self.max_eef_delta:
            eef_delta = move_direction / np.linalg.norm(move_direction, np.inf) * self.max_eef_delta
        else:
            eef_delta = move_direction
        return eef_delta
    def oracle_step(self):
        if self.use_push_primitive:
            return self.oracle_primitive_step()
        else:
            return self.oracle_delta_step()

    def collect_demonstrations(self, n_demonstrations: int, path: str):
        """
        Collects and stores demonstrations of the task using the oracle policy. The observation and action type
        are taken from the class instance. Demonstrations are stored as a pickle of a  List of ur-sim::Demonstration's


        :param n_demonstrations:
        :param path: path to store the pickle file
        :return: List of demonstrations
        """
        demonstrations = []
        for i in range(n_demonstrations):
            demonstration = Demonstration()
            obs = self.reset()
            done = False
            demonstration.observations.append(obs)
            while not done:
                action = self.oracle_step()
                obs,reward,done,_ = self.step(action)
                demonstration.actions.append(action)
                demonstration.observations.append(obs)
            demonstrations.append(demonstration)

        # store demonstrations in pickle file
        with open(path, "wb") as handle:
            pickle.dump(demonstrations, handle)

        return demonstrations

    def execute_primitive_oracle(self):
        """
        convenience function that calculates the optimal angle and length assuming perfect observation.
        """
        angle, length = self.oracle_primitive_step()
        self._execute_motion_primitive(angle, length)




    @staticmethod
    def _clip_target_position(eef_target_position: np.ndarray) -> np.ndarray:
        eef_target_position[0] = np.clip(
            eef_target_position[0], UR3ePush.eef_space_bounds[0], UR3ePush.eef_space_bounds[1]
        )
        eef_target_position[1] = np.clip(
            eef_target_position[1], UR3ePush.eef_space_bounds[2], UR3ePush.eef_space_bounds[3]
        )
        eef_target_position[2] = np.clip(
            eef_target_position[2], UR3ePush.eef_space_bounds[4], UR3ePush.eef_space_bounds[5]
        )
        return eef_target_position

    @staticmethod
    def _position_is_in_workspace(position: np.ndarray, margin: float = 0.0) -> bool:
        x, y = position[:2]

        if not (UR3ePush.eef_space_bounds[0] + margin < x < UR3ePush.eef_space_bounds[1] - margin):
            return False
        if not (UR3ePush.eef_space_bounds[2] + margin < y < UR3ePush.eef_space_bounds[3] - margin):
            return False

        if position.shape[0] == 3:
            z = position[2]
            if not (UR3ePush.eef_space_bounds[4] + margin < z < UR3ePush.eef_space_bounds[5] - margin):
                return False
        return True

    @staticmethod
    def position_is_in_object_space(position: np.ndarray, margin: float = 0.0) -> bool:
        return UR3ePush._position_is_in_workspace(
            position,
            margin=UR3ePush.object_radius + UR3ePush.robot_flange_radius + UR3ePush.robot_motion_margin + margin,
        )

    @staticmethod
    def get_random_object_position(goal_position: np.ndarray) -> np.ndarray:
        """
        Brute-force sample positions until one is in the allowed object workspace
        """
        while True:
            x = np.random.random() - 0.5
            y = np.random.random() * 0.5 - 0.5
            position = np.array([x, y])
            logging.debug(f"proposed object reset {position}")
            if (
                UR3ePush._position_is_in_workspace(
                    position,
                    margin=1.01
                    * (UR3ePush.object_radius + UR3ePush.robot_flange_radius + UR3ePush.robot_motion_margin),
                )
                and not np.linalg.norm(position - goal_position) < UR3ePush.goal_l2_margin
            ):
                return position

    def _get_random_eef_position(self) -> np.ndarray:
        while True:
            x = np.random.random() - 0.5
            y = np.random.random() * 0.5 - 0.5
            z = np.random.random() * 0.2
            position = np.array([x, y, z])
            logging.debug(f"proposed eef reset {position}")
            if UR3ePush._position_is_in_workspace(position):
                return position


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    env = UR3ePush(real_time=True, push_primitive=False, state_observation=False)
    done = True
    while True:
        if done:
            obs = env.reset()
        #angle, distance = env.oracle_primitive_step()
        # angle = np.random.random(1).item() * 2 * np.pi
        # distance = np.random.random(1).item() * 0.2
        action = env.oracle_step()
        print(action)
        obs, reward, done, _ = env.step(action)
        print(obs.shape)
        print(done)
        # env.render()
