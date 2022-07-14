from pathlib import Path
from typing import Union

import imageio.v2 as iio
import numpy as np
import wandb
from gym.core import Wrapper
from PIL import Image


class VideoRecorderWrapper(Wrapper):
    """
    A Gym Wrapper to log videos of the agent during training.
    Uses .gif's as this seemed to be most cross-platform and browser compatible.
    The SB3 VideoRecorder caused issues as it uses libx264 (H.264), which was not available. Furthermore it logs videos
    continuously instead of episode-based.

    Keep in mind that this is somewhat expensive (both in storage and CPU usage) so do not store too much videos.
    """

    def __init__(
        self, env, video_folder: Union[Path, str], capture_every_n_episodes: int = 10, log_wandb: bool = True
    ):
        super(VideoRecorderWrapper, self).__init__(env)

        self.frames = []
        self.episode_count = -1
        self.capture_period = capture_every_n_episodes

        self.video_path = Path(video_folder)

        self.log_wandb = log_wandb

        if self.log_wandb:
            pass

        # create folder if necessary
        self.video_path.mkdir(exist_ok=True, parents=True)

        # todo: make these configurable
        self.rescale_factor = 8  # downscale the rendered observations to reduce size of gifs
        self.num_black_frames_at_beginning = (
            10  # makes sure you can see when the episode starts if the video is playing in a loops
        )

    def reset(self):
        self.episode_count += 1

        if self._should_capture_this_episode():
            self.frames = []
            self._capture_current_frame()

        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self._should_capture_this_episode():
            self._capture_current_frame()
            if done:
                self._create_and_store_gif()
        return obs, reward, done, info

    def _should_capture_this_episode(self):
        return self.episode_count % self.capture_period == 0

    def _capture_current_frame(self):
        rgb = self.env.render(mode="rgb_array")
        rgb = Image.fromarray(rgb)
        rgb = rgb.resize((rgb.size[0] // self.rescale_factor, rgb.size[1] // self.rescale_factor))
        rgb = np.array(rgb)
        self.frames.append(rgb)

    def _create_and_store_gif(self):
        # compile
        gif_path = self.video_path / f"episode_{self.episode_count}.gif"

        frames = np.stack(
            # add black frames to mark beginning of episode (useful during playbacks)
            np.concatenate(
                (np.zeros((self.num_black_frames_at_beginning,) + self.frames[0].shape, dtype=np.uint8), self.frames)
            ),
            axis=0,
        )
        iio.mimsave(gif_path, frames)
        if self.log_wandb:
            wandb.log(
                {
                    "video": wandb.Video(str(gif_path), caption=f"episode_{self.episode_count}", format="gif"),
                    "episode": self.episode_count,
                }
            )
