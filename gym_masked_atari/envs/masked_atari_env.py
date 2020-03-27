import gym
from gym.envs import atari
import numpy as np
from .frostbite_masker import FROSTBITE_MASKER_DEFINITIONS


class MaskedAtariEnv(atari.AtariEnv):
    # TODO: what does this do?
    # metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        super(MaskedAtariEnv, self).__init__(**kwargs)
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
        self.zero_mask = np.ones((len(self.masker_definitions), *self.default_observation_shape), dtype=np.uint8)

        for i, masker_def in enumerate(self.masker_definitions):
            if zero_mask_indices is not None and i in zero_mask_indices:
                self.zero_mask[i, :, :] = 0

            elif masker_def.range_whitelist:
                self.zero_mask[i, :masker_def.row_range[0], :] = 0
                self.zero_mask[i, masker_def.row_range[1]:, :] = 0
                self.zero_mask[i, :, :masker_def.col_range[0]] = 0
                self.zero_mask[i, :, masker_def.col_range[1]:] = 0

            else:
                self.zero_mask[i, masker_def.row_range[0]:masker_def.row_range[1],
                               masker_def.col_range[0]:masker_def.col_range[1]] = 0

        self.observation_space.shape = self.zero_mask.shape

    def step(self, action):
        result = super(MaskedAtariEnv, self).step(action)
        obs = result[0]

        all_mask_results = (obs == self.all_colors).all(axis=3)
        category_masks = np.zeros((len(self.masker_definitions), *self.default_observation_shape), dtype=np.uint8)

        current_index = 0
        for i, length in enumerate(self.category_lengths):
            if length > 1:
                category_masks[i] = all_mask_results[current_index: current_index + length].any(axis=0)
                current_index += length
            else:
                category_masks[i:] = all_mask_results[current_index:]
                break

        return (category_masks * self.zero_mask,) + result[1:]
