# Pybullet playground for Robot Learning. 

The robot setup mimicks the AIRO research setup with UR3e robots, Robotiq 2F85 grippers and Zed2i cameras. 

## Pybullet Env 
### Local Installation
- create a conda environment `conda env create -n robot-learning-playground` and activate it with `conda activate robot-learning-playground`
- move into the `ur_sim` folder and run `pip install -r requirements.txt`
- manually install the `ur_ikfast` package using the instructions [here](https://github.com/cambel/ur_ikfast) (and test using `test_ikfast.py`)
- test that pybullet is working by running the `push_env.py` file. You should see the robot moving.

## Learning Algorithms
### Local Installation
- activate the conda environment
- from the learning folder run `conda env update -f learning.yml`