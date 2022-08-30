from fileinput import close
import math
from typing import List
import pybullet as p 
import pybullet_data
from pybullet_sim.assets.path import get_asset_root_folder
from pybullet_sim.pybullet_utils import HideOutput

class Gripper():
    def __init__(self, gripper_id, open_position, closed_position ) -> None:
        self.gripper_id = gripper_id
        self.open_position = open_position
        self.closed_position = closed_position
        self.reset()

    def reset(self, pose:List[float] = None):

        if pose is not None:
            p.resetBasePositionAndOrientation(self.gripper_id, pose[:3], pose[3:])

    def open_gripper(self,max_force: int = 100):
        self.movej(1.0,max_force)
    
    def close_gripper(self,max_force: int = 100):
        self.movej(0.0, max_force)

    def movej(self, position:float, max_force: int = 100):

        abs_position = self.closed_position + (self.open_position - self.closed_position) * position
        return self._movej(abs_position, max_force)
    
    def _movej(self, position:float, max_force: int):
        raise NotImplementedError

    def is_object_grasped(self):
        raise NotImplementedError
    


class Robotiq2F85(Gripper):
    def __init__(self) -> None:
        gripper_id = p.loadURDF(str(get_asset_root_folder()  / "robotiq2f85" / "robotiq_2f_85.urdf"),useFixedBase=False,flags=p.URDF_USE_INERTIA_FROM_FILE,
)
        open_position = 0.0
        closed_position = 0.085
        super().__init__(gripper_id,open_position,closed_position)

    def _movej(self, open_length,max_force):
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation approx      
        right_finger_dict = {5:1,7:-1,9:1} #outer-knuckle,inner-finger,inner-knuckle
        left_finger_dict = {0:1,2:-1,4:1} #outer-knuckle,inner-finger,inner-knuckle
        for finger_dict in [right_finger_dict, left_finger_dict]:
            for id, direction in finger_dict.items():
                p.setJointMotorControl2(self.gripper_id, id,p.POSITION_CONTROL,targetPosition=open_angle * direction,force=max_force, maxVelocity=0.5)

class WSG50(Gripper):
    def __init__(self):
        # values taken from https://colab.research.google.com/drive/1eXq-Tl3QKzmbXGSKU2hDk0u_EHdfKVd0?usp=sharing
        # and then adapted
        open_position = 0.0
        closed_position = 0.085
        self.home_joint_positions = [0.000000, -0.011130, -0.206421, 0.205143, -0.0, 0.000000, -0.0, 0.000000]
        with HideOutput():
            gripper_id = p.loadSDF("gripper/wsg50_one_motor_gripper_new_free_base.sdf")[0]
     
        super().__init__(gripper_id,open_position, closed_position)
        self.left_pincher_joint_id  = 4
        self.right_pincher_joint_id = 6

    
    def reset(self, pose=None):
        for jointIndex in range(p.getNumJoints(self.gripper_id)):
            p.resetJointState(self.gripper_id, jointIndex, self.home_joint_positions[jointIndex])
            p.setJointMotorControl2(self.gripper_id, jointIndex, p.POSITION_CONTROL, self.home_joint_positions[jointIndex], 0)
        super().reset(pose)


    def _movej(self, position, max_force):
        for id in [self.left_pincher_joint_id, self.right_pincher_joint_id]:
            p.setJointMotorControl2(self.gripper_id,id, p.POSITION_CONTROL, targetPosition=position, maxVelocity=0.5, force=max_force)




if __name__ == "__main__":
    import time 
    from pybullet_sim.hardware.ur3e import UR3e
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    #p.setGravity(0, 0, -9.81)
    tableId = p.loadURDF(str(get_asset_root_folder() / "ur3e_workspace" / "workspace.urdf"), [0, -0.3, -0.01])

    target = p.getDebugVisualizerCamera()[11]
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=target)



    robot = UR3e(simulate_real_time=True)
    gripper = Robotiq2F85()
    gripper.reset(robot.get_eef_pose())
    # print(robot.get_eef_pose())
    # print(p.getLinkState(gripper.gripper_id,0))

    # id = p.createVisualShape(p.GEOM_BOX)

    kuka_cid = p.createConstraint(robot.robot_id,robot.eef_id, gripper.gripper_id, 0, p.JOINT_FIXED, [0, 0, 0.0], [0.0, 0.0, 0.0], [0, 0,0.0])
    #gripper.movej(0.8,100)

    robot.movep([0.2,-0.0,0.2,0,0,0,1],speed=0.001)

    # for _ in range(1000):
    #     p.stepSimulation()
    #     time.sleep(1/240.0)
    time.sleep(100)
    p.disconnect()
