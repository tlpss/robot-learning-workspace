from ur_sim.push_env import UR3ePush
from pathlib import Path

def collect_demos():
    env = UR3ePush(state_observation=False, push_primitive=False,real_time=False)
    demos = env.collect_demonstrations(20,str(Path(__file__).parent / "push_demos.pkl"))


