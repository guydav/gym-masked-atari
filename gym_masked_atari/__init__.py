from gym.envs.registration import register

# Basically copied from the regular gym's registration code

obs_type = 'image'
nondeterministic = False

# # print ', '.join(["'{}'".format(name.split('.')[0]) for name in atari_py.list_games()])
for game in ['adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'defender', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']:

    # space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
    name = ''.join([g.capitalize() for g in game.split('_')])


    register(
        id='Masked{}-v0'.format(name),
        entry_point='gym_masked_atari.envs:MaskedAtariEnv',
        kwargs={'game': game, 'obs_type': obs_type, 'repeat_action_probability': 0.25},
        max_episode_steps=10000,
        nondeterministic=nondeterministic,
    )

    register(
        id='Masked{}-v4'.format(name),
        entry_point='gym_masked_atari.envs:MaskedAtariEnv',
        kwargs={'game': game, 'obs_type': obs_type},
        max_episode_steps=100000,
        nondeterministic=nondeterministic,
    )

    # Standard Deterministic (as in the original DeepMind paper)
    if game == 'space_invaders':
        frameskip = 3
    else:
        frameskip = 4

    # Use a deterministic frame skip.
    register(
        id='Masked{}Deterministic-v0'.format(name),
        entry_point='gym_masked_atari.envs:MaskedAtariEnv',
        kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip, 'repeat_action_probability': 0.25},
        max_episode_steps=100000,
        nondeterministic=nondeterministic,
    )

    register(
        id='Masked{}Deterministic-v4'.format(name),
        entry_point='gym_masked_atari.envs:MaskedAtariEnv',
        kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip},
        max_episode_steps=100000,
        nondeterministic=nondeterministic,
    )

    register(
        id='Masked{}NoFrameskip-v0'.format(name),
        entry_point='gym_masked_atari.envs:MaskedAtariEnv',
        kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1, 'repeat_action_probability': 0.25}, # A frameskip of 1 means we get every frame
        max_episode_steps=frameskip * 100000,
        nondeterministic=nondeterministic,
    )

    # No frameskip. (Atari has no entropy source, so these are
    # deterministic environments.)
    register(
        id='Masked{}NoFrameskip-v4'.format(name),
        entry_point='gym_masked_atari.envs:MaskedAtariEnv',
        kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1}, # A frameskip of 1 means we get every frame
        max_episode_steps=frameskip * 100000,
        nondeterministic=nondeterministic,
    )

