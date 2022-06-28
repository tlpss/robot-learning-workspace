import time

import mujoco
import numpy as np
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
import PIL.Image
from ur_sim.utils import get_assets_path
import matplotlib.pyplot as plt
from dm_control import mjcf
from dm_control import viewer

if __name__ == "__main__":
    ur3e_path = get_assets_path() / "ur3e"/ "ur3e.xml"


    arena = mjcf.RootElement()
    for x in [-2, 2]:
        arena.worldbody.add('light', pos=[x, -1, 3], dir=[-x, 1, -2])
    chequered = arena.asset.add('texture', type='2d', builtin='checker', width=300,
                                height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
    grid = arena.asset.add('material', name='grid', texture=chequered,
                           texrepeat=[5, 5], reflectance=.2)
    arena.worldbody.add('geom', type='plane', size=[2, 2, .1], material=grid)
    robot = mjcf.from_path(str(ur3e_path))

    arena.attach(robot)
    physics = mjcf.Physics.from_mjcf_model(arena)
    # Visualize the joint axis.
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True

    with physics.reset_context():
       print(physics.named.data.qpos)
       physics.named.data.qpos["ur3e/shoulder_pan_joint"] =  np.pi
       physics.named.data.qpos["ur3e/shoulder_lift_joint"] = - np.pi / 3


    for i in range(20):
        physics.step()
        pixels = physics.render(scene_option=scene_option)
        img = PIL.Image.fromarray(pixels)
        img.show()
        time.sleep(0.1)

