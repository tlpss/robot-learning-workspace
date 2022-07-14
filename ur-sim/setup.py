import setuptools

setuptools.setup(
    name="ur_sim",
    version="0.0.1",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    description="Mujoco Simulation Toolbox for UR robots at the AIRO reasearch group",
    packages=["ur_sim"],
    install_requires=["mujoco>=2.2.0", "dm-control==1.0.3.post1"],
)
