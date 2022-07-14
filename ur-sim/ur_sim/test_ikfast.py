import numpy as np

from ur_ikfast.ur_ikfast import ur_kinematics

if __name__ == "__main__":
    ur3e_arm = ur_kinematics.URKinematics("ur3e")

    """
    IKFast seems to struggle with quaternions along axis (wich leads to zeros..)
    also note that the order for the wrapper is w,x,y,z!
    """
    joint_angles = [-1.6, -1.6, 1.6, -1.6, -1.6, 0.0]  # in radians
    print("joint angles", joint_angles)
    pose_quat = ur3e_arm.forward(joint_angles)
    print("forward() quaternion \n", pose_quat)

    # print("inverse() all", ur3e_arm.inverse(pose_quat, True))
    pose_quat = [-0.2, -0.2, 0.2, 0.0, 1.0, 0.0, 0]
    print("inverse() one from quat", ur3e_arm.inverse(pose_quat, False, q_guess=joint_angles))

    pose_quat[4:] += np.random.randn(3) * 0.01
    print("inverse() one from noisy quat", ur3e_arm.inverse(pose_quat, False, q_guess=joint_angles))
