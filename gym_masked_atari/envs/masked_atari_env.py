from gym.envs import atari
import numpy as np
import cv2
from .frostbite_masker import FROSTBITE_MASKER_DEFINITIONS


INCLUDE_PIXELS_KEY = 'include_pixels'
GRAYSCALE_PIXELS_KEY = 'grayscale_pixels'


class MaskedAtariEnv(atari.AtariEnv):
    # TODO: what does this do?
    # metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        super(MaskedAtariEnv, self).__init__(**kwargs)
        self.include_pixels = True
        if INCLUDE_PIXELS_KEY in kwargs:
            self.include_pixels = kwargs[INCLUDE_PIXELS_KEY]

        self.grayscale_pixels = True
        if GRAYSCALE_PIXELS_KEY in kwargs:
            self.grayscale_pixels = kwargs[GRAYSCALE_PIXELS_KEY]

        self.default_observation_shape = self.observation_space.shape
        self.masker_definitions = None
        self.all_colors = None
        self.category_lengths = None
        self.zero_mask = None
        self.setup()

    def setup(self, masker_definitions=FROSTBITE_MASKER_DEFINITIONS, zero_mask_indices=None):
        if isinstance(masker_definitions, dict):
            masker_definitions = masker_definitions.values()

        self.masker_definitions = sorted(list(masker_definitions), key=lambda md: len(md.filter_colors), reverse=True)

        all_colors = []
        for masker in self.masker_definitions:
            all_colors.extend([np.array(c, dtype=np.uint8) for c in masker.filter_colors])

        self.all_colors = np.stack([np.ones(self.default_observation_shape, dtype=np.uint8) * c for c in all_colors])
        self.category_lengths = [len(masker.filter_colors) for masker in self.masker_definitions]
        self.zero_mask = np.ones((*self.default_observation_shape[:-1], len(self.masker_definitions)), dtype=np.uint8)

        for i, masker_def in enumerate(self.masker_definitions):
            if zero_mask_indices is not None and i in zero_mask_indices:
                self.zero_mask[:, :, i] = 0

            elif masker_def.range_whitelist:
                self.zero_mask[:masker_def.row_range[0], :, i] = 0
                self.zero_mask[masker_def.row_range[1]:, :, i] = 0
                self.zero_mask[:, :masker_def.col_range[0], i] = 0
                self.zero_mask[:, masker_def.col_range[1]:, i] = 0

            else:
                self.zero_mask[masker_def.row_range[0]:masker_def.row_range[1],
                               masker_def.col_range[0]:masker_def.col_range[1], i] = 0

        self.observation_space.shape = self.zero_mask.shape
        if self.include_pixels:
            increment = 1
            if not self.grayscale_pixels:
                increment = 3

            self.observation_space.shape = self.observation_space.shape[:-1] + (self.observation_space.shape[-1] + increment,)

    def reset(self, **kwargs):
        obs = super(MaskedAtariEnv, self).reset()
        return self._augment_observation(obs)

    def step(self, action):
        result = super(MaskedAtariEnv, self).step(action)
        return (self._augment_observation(result[0]),) + result[1:]

    def _augment_observation(self, obs):
        all_mask_results = np.moveaxis((obs == self.all_colors).all(axis=3), 0, -1)
        # print(all_mask_results.shape)
        category_masks = np.zeros((*self.default_observation_shape[:-1], len(self.masker_definitions)), dtype=np.uint8)
        # print(category_masks.shape)

        current_index = 0
        for i, length in enumerate(self.category_lengths):
            if length > 1:
                # print(category_masks[i].shape)
                # print(all_mask_results[current_index: current_index + length].any(axis=0).shape)
                category_masks[:, :, i] = all_mask_results[:, :, current_index: current_index + length].any(axis=-1)
                current_index += length
            else:
                category_masks[:, :, i:] = all_mask_results[:, :, current_index:]
                break

        masks = category_masks * self.zero_mask

        if self.include_pixels:
            if self.grayscale_pixels:
                obs = np.expand_dims(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), -1)

            return np.concatenate((obs, masks), -1)

        else:
            return masks
