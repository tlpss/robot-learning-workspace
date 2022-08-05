import logging
import time

import numpy as np
import pybullet
import pybullet as p
import pybullet_data
from ur_sim.assets.path import get_asset_root_folder
from ur_sim.pybullet_utils import HideOutput

asset_path = get_asset_root_folder()

from ur_ikfast import ur_kinematics


class UR3e:
    """
    Synchronous Interface for a simulated UR3e robot in PyBullet.
    Uses the (limited) inverse kinematics of Bullet.
    """

    def __init__(self, robot_base_position=None, eef_start_pose=None, simulate_real_time=False):
        self.simulate_real_time = simulate_real_time
        self.homej = np.array([-0.5, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        if robot_base_position is None:
            self.robot_base_position = [0, 0, 0]
        else:
            assert len(robot_base_position) == 3
            self.robot_base_position = robot_base_position

        self.robot_id = None
        self.joint_ids = None
        self.eef_id = 9  # manually determined

        self.ikfast_ur3e_solver = ur_kinematics.URKinematics("ur3e")

        self.reset(eef_start_pose)

    def reset(self, pose=None):
        with HideOutput():
            self.robot_id = p.loadURDF(
                str(asset_path / "ur3e" / "ur3e.urdf"),
                self.robot_base_position,
                flags=pybullet.URDF_USE_INERTIA_FROM_FILE,
            )

        # Get revolute joint indices of robot_id (skip fixed joints).
        n_joints = p.getNumJoints(self.robot_id)
        joints = [p.getJointInfo(self.robot_id, i) for i in range(n_joints)]
        self.joint_ids = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        # Move robot_id to home joint configuration.
        for i in range(len(self.joint_ids)):
            p.resetJointState(self.robot_id, self.joint_ids[i], self.homej[i])

        if pose is not None:
            joint_config = self.solve_ik(pose)
            # Move robot to target pose starting from the home joint configuration.
            for i in range(len(self.joint_ids)):
                p.resetJointState(self.robot_id, self.joint_ids[i], joint_config[i])

    def get_eef_pose(self) -> np.ndarray:
        """

        :return: 7D pose: Position [m], Orientation [Quaternion, radians]
        """
        link_info = p.getLinkState(self.robot_id, self.eef_id)
        position = link_info[0]
        orientation = link_info[1]
        return np.array(position + orientation)

    def get_joint_configuration(self) -> np.ndarray:
        """

        :return: 6D joint configuration in radians
        """
        return np.array([p.getJointState(self.robot_id, i)[0] for i in self.joint_ids])

    def movej(self, targj: np.ndarray, speed=0.01, max_steps: int = 100) -> bool:
        """Move UR5 to target joint configuration.
        adapted from https://github.com/google-research/ravens/blob/d11b3e6d35be0bd9811cfb5c222695ebaf17d28a/ravens/environments/environment.py#L351

        :param targj: A 6D np.ndarray with target joint configuration
        :param speed: max joint velocity
        :param timeout: max execution time for the robot to get there
        :return: True if the robot was able to move to the target joint configuration
        """
        for step in range(max_steps):
            currj = self.get_joint_configuration()
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return True

            # Move with constant velocity by setting target joint configuration
            # to an intermediate configuration at each step
            # This also somehow makes motions more linear in the EEF space #todo: figure out why..
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            for id, joint in enumerate(self.joint_ids):
                # set individual to allow for setting the max Velocity
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=self.joint_ids[id],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=stepj[id],
                    velocityGain=0.5,
                    positionGain=2.0,
                    # maxVelocity=2.0,
                    force=100,  # this is way too high for the UR3e, but makes the simulation more stable..
                )

            # self._compensate_gravity()

            p.stepSimulation()
            if self.simulate_real_time:
                time.sleep(1.0 / 240)
        logging.debug(f"Warning: movej exceeded {max_steps} simulation steps. Skipping.")
        return False

    def movep(self, pose: np.ndarray, speed=0.01, max_steps=1000) -> bool:
        """Move UR3e to target end effector pose.

        :param pose: a 7D pose - position [meters] + orientation [Quaternion, radians]
        """
        targj = self.solve_ik_ikfast(pose)
        return self.movej(targj, speed, max_steps)

    def movep_linear(self, pose: np.ndarray, speed=1.5):
        # todo: linear EEF motions
        raise NotImplementedError

    def solve_ik_ikfast(self, pose: np.ndarray) -> np.ndarray:
        p = np.copy(pose)
        w = pose[-1]
        p[4:] = pose[3:6]
        p[3] = w

        targj = None
        for _ in range(6):
            # fix for axis-aligned orientations (which is often used in e.g. top-down EEF orientations
            # add random noise to the EEF orienation to avoid axis-alignment
            # see https://github.com/cambel/ur_ikfast/issues/4
            p[4:] += np.random.randn(3) * 0.01
            targj = self.ikfast_ur3e_solver.inverse(p, q_guess=self.get_joint_configuration())
            if targj is not None:
                break

        if targj is None:
            raise ValueError("IKFast failed... most likely the pose is out of reach of the robot?")
        return targj

    def solve_ik(self, pose: np.ndarray) -> np.ndarray:
        """Calculate joint configuration with inverse kinematics.

        :param pose: a 7D pose - position [meters] + orientation [Quaternion, radians]

        :return a 6D joint configuration
        """
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.eef_id,
            targetPosition=pose[0:3],
            targetOrientation=pose[3:7],
            # tuned these params to avoid cray IK solutions.. especially shoulder,elbow and wrist 3 are important!
            # note that these limit the workspace of the robot!
            # with a decent planner such constraints are not required nor desirable.
            # also note that unreachable poses will still return a configuration...
            lowerLimits=[-5 / 4 * np.pi, -np.pi, 0, -1.2 * np.pi, -np.pi, -2 * np.pi],
            upperLimits=[2 * np.pi / 4, 0, np.pi, -0.2 * np.pi, 0, 2 * np.pi],
            jointRanges=[1.5 * np.pi, np.pi, np.pi, np.pi, np.pi, 4 * np.pi],
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def _compensate_gravity(self):
        link_masses = [2.0, 2.0, 3.42, 1.26, 0.8, 0.8, 0.35]
        for id, mass in zip(self.joint_ids, link_masses):
            p.applyExternalForce(self.robot_id, id, forceObj=[0, 0, 9.81 * mass], posObj=[0, 0, 0], flags=p.LINK_FRAME)


if __name__ == "__main__":
    """
    simple script to move the UR3e
    """
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -9.81)

    target = p.getDebugVisualizerCamera()[11]
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=target)

    planeId = p.loadURDF("plane.urdf", [0, 0, -1.0])
    tableId = p.loadURDF(str(asset_path / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.01])
    robot = UR3e(simulate_real_time=True)
    time.sleep(2.0)
    for _ in range(200):
        p.stepSimulation()
    pose = [0.4, -0.0, 0.1]
    pose.extend(p.getQuaternionFromEuler([np.pi, 0, 0]))
    pose = np.array(pose)
    robot.movep(pose, max_steps=2000)
    pose = [0.3, -0.4, 0.1]
    pose.extend(p.getQuaternionFromEuler([np.pi, 0, 0]))
    pose = np.array(pose)
    robot.movep(pose, max_steps=2000)
    pose = [-0.2, -0.4, 0.1]
    pose.extend(p.getQuaternionFromEuler([np.pi, 0, 0]))
    pose = np.array(pose)
    robot.movep(pose, max_steps=2000)
    pose = [-0.2, -0.23, 0.2]
    pose.extend(p.getQuaternionFromEuler([np.pi * 0.99, 0, 0]))
    pose = np.array(pose)
    robot.movep(pose, max_steps=2000)
    pose = [-0.4, -0.0, 0.01]
    pose.extend(p.getQuaternionFromEuler([np.pi, 0, 0]))
    pose = np.array(pose)
    robot.movep(pose, max_steps=2000)
    pose = [-0.0, -0.25, 0.01]
    pose.extend(p.getQuaternionFromEuler([np.pi, 0, 0]))
    pose = np.array(pose)
    robot.movep(pose, max_steps=2000)
    pose = [-0.0, -0.27, 0.01]
    pose.extend(p.getQuaternionFromEuler([np.pi, 0, 0]))
    pose = np.array(pose)
    robot.movep(pose, max_steps=2000)
    pose = [-0.0, -0.25, 0.0001]
    pose.extend(p.getQuaternionFromEuler([np.pi, 0, 0]))
    pose = np.array(pose)
    robot.movep(pose, max_steps=2000)
    time.sleep(10.0)
    p.disconnect()
