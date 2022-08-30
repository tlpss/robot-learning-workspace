import time

import pybullet as p
import pybullet_data

if __name__ == "__main__":

    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")
    startPos = [0, 0, 0.5]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0.0])
    urId = p.loadURDF("assets/ur3e/ur3e.urdf", startPos, startOrientation)
    print(p.getNumJoints(urId))
    # set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
    for i in range(10000):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
    p.disconnect()
