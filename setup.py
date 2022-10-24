import setuptools

setuptools.setup(
    name="robot_learning",
    version="0.0.1",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    description="playground for robot (policy) learning experiments",
    packages=["robot_learning"],
    install_requires=[
        "torch",
        "wandb",
        "torchvision",
        "stable-baselines3",
        "pytorch-lightning",
        "timm",
        "pytest",
        "pre-commit",
    ],
)
