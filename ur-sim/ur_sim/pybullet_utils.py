import pybullet as p


def set_debug_rendering(state: int):
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, state)


def disable_debug_rendering():
    """
    Should be used during importing of URDFs for increased loading speed.
    :return:
    """
    set_debug_rendering(0)


def enable_debug_rendering():
    set_debug_rendering(1)
