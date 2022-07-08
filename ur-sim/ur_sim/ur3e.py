from typing import List, Tuple

import matplotlib.pyplot as plt
import pybullet
import pybullet as p
import pybullet_data
from ur_sim.assets.path import get_asset_root_folder
import numpy as np
import time
asset_path = get_asset_root_folder()


class UR3e:
    """
    Class for creating and interacting with a UR3e robot in PyBullet.
    """
    def __init__(self, robot_base_position = None):
        self.homej = np.array([-0.5, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        if robot_base_position is None:
            self.robot_base_position = [0, 0, 0]
        else:
            assert len(robot_base_position) == 3
            self.robot_base_position = robot_base_position

        self.robot_id = None
        self.joint_ids = None
        self.eef_id = 9  # manually determined
        self.reset()

    def reset(self):
        self.robot_id = p.loadURDF(str(asset_path / "ur3e" / "ur3e.urdf"), self.robot_base_position, flags=pybullet.URDF_USE_SELF_COLLISION)

        # Get revolute joint indices of robot_id (skip fixed joints).
        n_joints = p.getNumJoints(self.robot_id)
        joints = [p.getJointInfo(self.robot_id, i) for i in range(n_joints)]
        self.joint_ids = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        # Move robot_id to home joint configuration.
        for i in range(len(self.joint_ids)):
            p.resetJointState(self.robot_id, self.joint_ids[i], self.homej[i])

    def get_eef_pose(self) -> Tuple[List,List]:
        """

        :return: Position [m], Orientation [Quaternion, radians]
        """
        link_info = p.getLinkState(self.robot_id, self.eef_id,)
        position = link_info[0]
        orientation = link_info[1]
        return position, orientation

    def get_joint_configuration(self) -> List:
        """

        :return: 6D joint configuration in radians
        """
        return [p.getJointState(self.robot_id, i)[0] for i in self.joint_ids]

    def movej(self, targj: List, speed=0.01, timeout=10) -> bool:
        """Move UR5 to target joint configuration.
        adapted from https://github.com/google-research/ravens/blob/d11b3e6d35be0bd9811cfb5c222695ebaf17d28a/ravens/environments/environment.py#L351

        :param targj: A 6D list with target joint configuration
        :param speed: max joint velocity
        :param timeout: max execution time for the robot to get there
        :return: True if the robot was able to move to the target joint configuration
        """
        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = self.get_joint_configuration()
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return True

            # Move with constant velocity by setting target joint configuration
            # to an intermediate configuration at each step
            norm = np.linalg.norm(diffj,ord=1)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joint_ids))
            p.setJointMotorControlArray(
                bodyIndex=self.robot_id,
                jointIndices=self.joint_ids,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            p.stepSimulation()
            time.sleep(1.0/240)
        print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        return False

    def movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""
        targj = self.solve_ik(pose)
        return self.movej(targj, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.eef_id,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints




if __name__ == "__main__":
    """
    simple script to move the UR3e 
    """
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)

    target = p.getDebugVisualizerCamera()[11]
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

    p.resetDebugVisualizerCamera(
        cameraDistance=1.8,
        cameraYaw=0,
        cameraPitch=-45,
        cameraTargetPosition=target)

    planeId = p.loadURDF("plane.urdf",[0,0,-1.0])
    tableId = p.loadURDF(str(asset_path / "ur3e_workspace" / "workspace.urdf"),[0,-0.3,-0.01])
    robot = UR3e()
    time.sleep(2.0)
    robot.movep(([0.2,-0.2,0.2],p.getQuaternionFromEuler([np.pi, 0,0])))
    # robot.movep(([-0.2,0.2,0.2],p.getQuaternionFromEuler([np.pi, 0,0])))
    # time.sleep(2.0)
    robot.movep(([-0.0,-0.23,0.2],p.getQuaternionFromEuler([np.pi, 0,0])))
    robot.movep(([-0.0,-0.23,0.1],p.getQuaternionFromEuler([np.pi, 0,0])))
    robot.movep(([-0.0,-0.23,0.01],p.getQuaternionFromEuler([np.pi, 0,0])))

    time.sleep(100.0)
    p.disconnect()




