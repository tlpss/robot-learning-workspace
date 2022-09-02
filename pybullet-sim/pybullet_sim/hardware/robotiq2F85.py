from fileinput import close
import time
import logging
import math
from typing import List
import pybullet as p 
import pybullet_data
from pybullet_sim.assets.path import get_asset_root_folder
from pybullet_sim.pybullet_utils import HideOutput
import math
import numpy as np

class Gripper():
    open_relative_position = 0.0
    closed_relative_position = 1.0

    def __init__(self, gripper_id,tcp_offset:np.ndarray, simulate_realtime: bool = True) -> None:
        self.gripper_id = gripper_id
        self.simulate_real_time = simulate_realtime
        self.tcp_offset = tcp_offset
        self.target_relative_position = Gripper.open_relative_position
        self.reset()

    def reset(self, pose:List[float] = None):
        self.target_relative_position = Gripper.open_relative_position
        self._set_joint_targets(self.target_relative_position, max_force=100)
        if pose is not None:
            p.resetBasePositionAndOrientation(self.gripper_id, pose[:3], pose[3:])

    def open_gripper(self,max_force: int = 100):
        self.movej(Gripper.open_relative_position,max_force)
    
    def close_gripper(self,max_force: int = 100):
        self.movej(Gripper.closed_relative_position, max_force)

    def movej(self, target_relative_position:float, max_force: int = 100, max_steps:int = 250):
        # bookkeeping
        self.target_relative_position = target_relative_position


        for _ in range(max_steps):
            current_relative_position = self.get_relative_position()
            if abs(target_relative_position - current_relative_position) < 3e-2:
                return True
            self._set_joint_targets(target_relative_position, max_force)
            p.stepSimulation()
            if self.simulate_real_time:
                time.sleep(1.0 / 240)
        logging.debug(f"Warning: movej exceeded {max_steps} simulation steps for {self.__class__}. Skipping.")
        
    
    def is_object_grasped(self):
        # rather hacky proxy, use with care..
        return abs(self.target_relative_position - self.get_relative_position()) > 0.05
    
    def attach_with_constraint_to_robot(self, robot_id, robot_link_id):
        # create the constraint to attach to the robot frame
        raise NotImplementedError
    def _set_joint_targets(self, target_relative_position:float, max_force: int):
        raise NotImplementedError

    def get_relative_position(self):
        raise NotImplementedError


class WSG50(Gripper):

    tcp_offset = np.array([0.0,0.0,0.153])
    closed_position = 0.050
    open_position = 0.0
    def __init__(self, simulate_realtime: bool = True) -> None:
        gripper_id = p.loadURDF(str(get_asset_root_folder() / "wsg50"/ "wsg50.urdf"))
        super().__init__(gripper_id, WSG50.tcp_offset, simulate_realtime)
        self._create_constraints()

    def _create_constraints(self):
        c = p.createConstraint(self.gripper_id, 0,
                            self.gripper_id, 2,
                            jointType=p.JOINT_GEAR,
                            jointAxis=[1,0, 0],
                            parentFramePosition=[0, 0, 0],
                            childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, maxForce=50, erp=0.1)  # Note: the mysterious `erp` is of EXTREME importance
    
    def _set_joint_targets(self, target_relative_position ,max_force):
        open_angle = self._relative_position_to_joint_opening(target_relative_position)
        for id in [0,2]:
            p.setJointMotorControl2(self.gripper_id, id,p.POSITION_CONTROL,targetPosition=open_angle,force=max_force, maxVelocity=0.1)

    def get_relative_position(self):
        joint_config = p.getJointState(self.gripper_id, 0)[0]
        return self._joint_opening_to_relative_position(joint_config)

    def attach_with_constraint_to_robot(self, robot_id, robot_link_id):
        p.createConstraint(robot_id,robot_link_id, self.gripper_id, -1, p.JOINT_FIXED, [0, 0, 0.0], [0.0, 0.0, 0], [0, 0,-0.005])
    @staticmethod
    def _joint_opening_to_relative_position(abs_position:float) -> float:
        rel_position = (abs_position - WSG50.open_position) / (WSG50.closed_position-WSG50.open_position)
        return rel_position

    @staticmethod
    def _relative_position_to_joint_opening(relative_position: float) -> float:
        abs_position = WSG50.open_position + (WSG50.closed_position - WSG50.open_position) * relative_position
        return abs_position
class Robotiq2F85(Gripper):
    """
    the Robotiq grippers are a pain to simulate as they have a closed chain due to their parallel inner and outer knuckles.
    Actuating the 6 joints seperately is not recommended as the joints would close faster/slower, resulting in unrealistic grasping. 
    In fact all joints on each finger (3/finger) should mimic each other, and so do the 2 fingers.

    The resulting physics are rather unstable however.. Grasping an object for example is very tricky..
    
    """
    open_position = 0.000
    closed_position = 0.085
    tcp_offset = np.array([0.0,0.0,0.15])

    def __init__(self) -> None:
        gripper_id = p.loadURDF(str(get_asset_root_folder()  / "robotiq2f85" / "robotiq_2f_85.urdf"),useFixedBase=False)
  
        super().__init__(gripper_id,tcp_offset=Robotiq2F85.tcp_offset)
        self._create_constraints()

    def _create_constraints(self):
        constraint_dict = {5:{7:-1,9:1},0: {2:-1,5:1,4:1}} # attach finger joint to outer knuckle to keep fingertips vertical.
        for parent_id, children_dict in constraint_dict.items():
            for joint_id, multiplier in children_dict.items():
                c = p.createConstraint(self.gripper_id, parent_id,
                                    self.gripper_id, joint_id,
                                    jointType=p.JOINT_GEAR,
                                    jointAxis=[0, 1, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=[0, 0, 0])
                p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance
            
    def _set_joint_targets(self, target_relative_position ,max_force):
        open_angle = self._relative_position_to_joint_angle(target_relative_position)

        # still actuate all joints to improve stability? (weird pybullet behavior if you don't)
        # thanks to constraints they will still be more or less in sync.

        right_finger_dict = {7:-1,9:1,5:1} 
        left_finger_dict = {0:1,2:-1,4:1} 
        for finger_dict in [right_finger_dict, left_finger_dict]:
            for id, direction in finger_dict.items():
                p.setJointMotorControl2(self.gripper_id, id,p.POSITION_CONTROL,targetPosition=open_angle * direction,force=100, maxVelocity=0.8)

    @staticmethod
    def _joint_angle_to_relative_position(angle:float) -> float:
        abs_position = math.sin(0.715-angle) * 0.1143 + 0.01
        rel_position = (abs_position - Robotiq2F85.closed_position) / (Robotiq2F85.open_position-Robotiq2F85.closed_position)
        return rel_position

    @staticmethod
    def _relative_position_to_joint_angle(relative_position: float) -> float:
        abs_position = Robotiq2F85.closed_position + (Robotiq2F85.open_position - Robotiq2F85.closed_position) * relative_position
        open_angle = 0.715 - math.asin((abs_position - 0.010) / 0.1143)  # angle calculation approx 
        return open_angle

    def get_relative_position(self):
        joint_config = p.getJointState(self.gripper_id, 0)[0]
        return self._joint_angle_to_relative_position(joint_config)

    def attach_with_constraint_to_robot(self, robot_id, robot_link_id):
        p.createConstraint(robot_id,robot_link_id, self.gripper_id, -1, p.JOINT_FIXED, [0, 0, 0.0], [0.0, 0.0, 0], [0, 0,-0.02],childFrameOrientation=p.getQuaternionFromEuler([0,0,1.57]))

if __name__ == "__main__":
    from pybullet_sim.hardware.ur3e import UR3e
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -10)
    tableId = p.loadURDF(str(get_asset_root_folder() / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.001])

    target = p.getDebugVisualizerCamera()[11]
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=target)
    gripper1 = WSG50()
    robot1 = UR3e(simulate_real_time=True,gripper=gripper1,eef_start_pose=np.array([0.2,-0.1,0.3,1,0,0,0]))


    gripper2 = Robotiq2F85()
    robot2 = UR3e(simulate_real_time=True,robot_base_position=[0.5,0.0,0.0],gripper=gripper2)

    for i in range(p.getNumJoints(gripper1.gripper_id)):
        print(p.getJointInfo(gripper1.gripper_id, i))


    gripper1.close_gripper()
    gripper2.close_gripper()

    robot1.movep([0.2,-0.2,0.0,1,0,0,0],speed=0.001)
    gripper1.open_gripper()
    gripper1.close_gripper()

    robot2.movep([0.2,-0.2,0.2,1,0,0,0],speed=0.001) #TODO: fix bug w/ base_position for robots.
    gripper2.open_gripper()
    gripper2.close_gripper()
    time.sleep(100)
    p.disconnect()
