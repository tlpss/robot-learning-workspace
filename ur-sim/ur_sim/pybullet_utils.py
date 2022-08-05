import os
import sys

import pybullet as p


class HideOutput(object):
    # Taken from Pybullet Tools at https://github.com/caelan/pybullet-planning/tree/master/pybullet_tools

    # https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
    # https://stackoverflow.com/questions/4178614/suppressing-output-of-module-calling-outside-library
    # https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
    """
    A context manager that block stdout for its scope, usage:
    with HideOutput():
        os.system('ls -l')
    """
    DEFAULT_ENABLE = True

    def __init__(self, enable=None):
        if enable is None:
            enable = self.DEFAULT_ENABLE
        self.enable = enable
        if not self.enable:
            return
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        if not self.enable:
            return
        self.fd = 1
        # self.fd = sys.stdout.fileno()
        self._newstdout = os.dup(self.fd)
        os.dup2(self._devnull, self.fd)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return
        sys.stdout.close()
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, self.fd)
        os.close(self._oldstdout_fno)  # Added


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
