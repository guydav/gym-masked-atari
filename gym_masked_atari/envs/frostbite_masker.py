from collections import namedtuple


MaskerDefinition = namedtuple('MaskerDefinition', ('name', 'filter_colors', 'row_range',
                                                   'col_range', 'range_whitelist'))
MaskerDefinition.__new__.__defaults__ = ('', None, (None, None), (None, None), False)


FULL_FRAME_SHAPE = (210, 160)
SMALL_FRAME_SHAPE = (84, 84)

player_colors = ((162, 98, 33),
                 (162, 162, 42),
                 (198, 108, 58),
                 (142, 142, 142)  # also captures the igloo
                )

unvisited_floe_colors = ((214, 214, 214),  # unvisited_floes
                        )
visited_floe_colors = ((84, 138, 210),  # visited floes
                      )
land_colors = ((192, 192, 192), # the lighter ground in earlier levels
               (74, 74, 74),  # the darker ground in later levels
              )

land_row_min = 42
land_row_max = 78
land_row_range = (land_row_min, land_row_max)

bad_animal_colors = ((132, 144, 252),  # birds -- also captures the score!
                     (213, 130, 74),  # crabs -- no more conflict with the player
                     (210, 210, 64),  # angry yellow things
                    )

bear_colors = ((111, 111, 111),  # bear in white background
               (214, 214, 214),  # bear in black background -- same as the unvisited floes
              )

good_animal_colors = ((111, 210, 111), # fish
                     )

animal_full_frame_row_min = 78
animal_full_frame_row_max = 185
animal_full_frame_row_range = (animal_full_frame_row_min, animal_full_frame_row_max)

igloo_colors = ((142, 142, 142),  # isolating the igloo door is harder - its black and orange colors both conflict
                )

igloo_full_frame_row_min = 35
igloo_full_frame_row_max = 55
igloo_full_frame_row_range = (igloo_full_frame_row_min, igloo_full_frame_row_max)

igloo_full_frame_col_min = 112
igloo_full_frame_col_max = 144
igloo_full_frame_col_range = (igloo_full_frame_col_min, igloo_full_frame_col_max)


AGENT_KEY = 'agent'
UNVISITED_FLOE_KEY = 'unvisited_floe'
VISITED_FLOE_KEY = 'visited_floe'
LAND_KEY = 'land'
BAD_ANIMAL_KEY = 'bad_animal'
GOOD_ANIMAL_KEY = 'good_animal'
BEAR_KEY = 'bear'
IGLOO_KEY = 'igloo'

agent_masker_def = MaskerDefinition(AGENT_KEY, player_colors, igloo_full_frame_row_range,
                                    igloo_full_frame_col_range)
unvisited_floe_masker_def = MaskerDefinition(UNVISITED_FLOE_KEY, unvisited_floe_colors, animal_full_frame_row_range,
                                             (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
visited_floe_masker_def = MaskerDefinition(VISITED_FLOE_KEY, visited_floe_colors, animal_full_frame_row_range,
                                           (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
land_masker_def = MaskerDefinition(LAND_KEY, land_colors, land_row_range,
                                   (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
bad_animal_masker_def = MaskerDefinition(BAD_ANIMAL_KEY, bad_animal_colors, animal_full_frame_row_range,
                                         (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
good_animal_masker_def = MaskerDefinition(GOOD_ANIMAL_KEY, good_animal_colors, animal_full_frame_row_range,
                                          (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
bear_filter_def = MaskerDefinition(BEAR_KEY, bear_colors, (0, animal_full_frame_row_min),
                                   (0, FULL_FRAME_SHAPE[1]), range_whitelist=True)
igloo_masker_def = MaskerDefinition(IGLOO_KEY, igloo_colors, igloo_full_frame_row_range,
                                    igloo_full_frame_col_range, range_whitelist=True)

FROSTBITE_MASKER_DEFINITIONS = {
    AGENT_KEY: agent_masker_def,
    UNVISITED_FLOE_KEY: unvisited_floe_masker_def,
    VISITED_FLOE_KEY: visited_floe_masker_def,
    LAND_KEY: land_masker_def,
    BAD_ANIMAL_KEY: bad_animal_masker_def,
    GOOD_ANIMAL_KEY: good_animal_masker_def,
    BEAR_KEY: bear_filter_def,
    IGLOO_KEY: igloo_masker_def
}


