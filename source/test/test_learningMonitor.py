# !/usr/bin/env python
import os
import sys

import pytest

import gym
from baselines import deepq
import uuid

from experiment_directory_tools import create_run_directory, clean_result_directory
from source.learning_monitor import LearningMonitor, learning_monitor_builder
from source.monitoring.stats_recorder import EpisodeDataClass, LearningDataCollector, LearningStatRecorder


MOCK_DIR = "./mock_up_directory"

CURRENT = "/current_run"
PAST = "/past_run"

INITIAL_WORKING_DIRECTORY = os.getcwd()


def set_up(run_dir):
    env = gym.make("LunarLander-v2")

    model = deepq.models.mlp([10, 10], layer_norm=True)

    dir = MOCK_DIR + CURRENT + run_dir

    return env, model, dir


def teardown():
    pass


@pytest.fixture(scope="function")
def prep_test():
    clean_result_directory(MOCK_DIR)
    run_dir = create_run_directory(MOCK_DIR, "unit_test", UUID=uuid.uuid4())
    yield set_up(run_dir)


rl_agent_config = {
    "experiment_name": "TEST DQN FalconX Red",
    "result_folder": MOCK_DIR + CURRENT,
    "run_dir": None,
    "experiment_file_name": None,

    "callback_class": None,
    "print_freq": 100,
    "checkpoint_freq": 1000,
    # "checkpoint_path":None,
    "visualize_episode_every": 400,
    "render_and_plot_data": 30,

    # === RL model ===
    "environment_name": "RocketLanderRed-v1",
    "VCS_tag": None,
    "observation_space_shape label": None,
    # "state variable label": ["x pos", "y pos", "angle", "left ctc",
    #                          "right ctc", "throttle", "eng. gimbal",
    #                          "x velocity", "y velocity", "ang velocity"
    #                          ],
    "state variable label": ["x pos", "y pos", "angle", "left ctc",
                             "right ctc",
                             "x velocity", "y velocity", "ang velocity"
                             ],

    "gamma": 0.9995,
    "param_noise": True,

    # === Model ===
    # MLP numbers of cells per layers
    "hiddens": [5],
    "layer_norm": False,

    # === Optimization ===
    "lr": 5e-4,
    "max_timesteps": 60,
    "learning_starts": 510,
    "train_freq": 1,
    "batch_size": 10,

    # === Replay buffer ===
    "buffer_size": 500,
    "prioritized_replay": False,
    "prioritized_replay_alpha": 0.6,
    "prioritized_replay_beta0": 0.4,
    "prioritized_replay_beta_iters": None,
    "prioritized_replay_eps": 1e-6,

    # === Exploration ===
    "exploration_fraction": 0.4,
    "exploration_final_eps": 0.01,
    "target_network_update_freq": 100,
}


def test_learning_monitor_convenience_builder_fct(prep_test):
    (env, model, dir) = prep_test

    """ ------- gym env wrapper ------------------------------------------------------------------------------------ """
    learning_monitor_config = {'collect_ep_data_every_callable': False}

    env, learning_data_collector = learning_monitor_builder(env, dir, learning_monitor_config, force=False)

    assert isinstance(env, gym.Env)
    assert isinstance(learning_data_collector, LearningDataCollector)


def test_learning_monitor_alt_constructor(prep_test):
    (env, model, dir) = prep_test

    """ ------- gym env wrapper ------------------------------------------------------------------------------------ """
    learning_monitor_config = {'collect_ep_data_every_callable': False}

    env, learning_data_collector = LearningMonitor.build_collectors(env, dir, learning_monitor_config, force=False)

    assert isinstance(env, gym.Env)
    assert isinstance(learning_data_collector, LearningDataCollector)


def test_learning_monitor(prep_test):
    (env, model, dir) = prep_test

    """ ------- gym env wrapper ------------------------------------------------------------------------------------ """
    learning_monitor_config = {
        # 'collect_ep_data_every_callable': lambda idx: False
        'collect_ep_data_every_callable': False
    }

    env, learning_data_collector = LearningMonitor.build_collectors(env, dir, learning_monitor_config, force=False)

    """ ------- learning wrapper ----------------------------------------------------------------------------------- """

    # TODO: remove and let the returned  'learning_data_collector' object from 'learning_monitor_builder' do is job
    # note: temporary fix
    def learning_data_collector(lcl, _glb):
        render = getattr(lcl['env'], 'render')
        if lcl['t'] >= rl_agent_config['max_timesteps'] + 1:
            render()
        return None

    """ ------- start learning ------------------------------------------------------------------------------------- """
    act = deepq.learn(env,
                      q_func=model,
                      lr=rl_agent_config['lr'],
                      max_timesteps=rl_agent_config['max_timesteps'],
                      buffer_size=rl_agent_config['buffer_size'],
                      exploration_fraction=rl_agent_config['exploration_fraction'],
                      exploration_final_eps=rl_agent_config['exploration_final_eps'],

                      train_freq=rl_agent_config['train_freq'],
                      batch_size=rl_agent_config['batch_size'],
                      print_freq=rl_agent_config['print_freq'],

                      checkpoint_freq=rl_agent_config['checkpoint_freq'],
                      # checkpoint_path=rl_agent_config['checkpoint_path'],
                      learning_starts=rl_agent_config['learning_starts'],
                      gamma=rl_agent_config['gamma'],
                      target_network_update_freq=rl_agent_config['target_network_update_freq'],
                      prioritized_replay=rl_agent_config['prioritized_replay'],
                      prioritized_replay_alpha=rl_agent_config['prioritized_replay_alpha'],
                      prioritized_replay_beta0=rl_agent_config['prioritized_replay_beta0'],
                      prioritized_replay_beta_iters=rl_agent_config['prioritized_replay_beta_iters'],
                      prioritized_replay_eps=rl_agent_config['prioritized_replay_eps'],
                      param_noise=rl_agent_config['param_noise'],
                      callback=learning_data_collector
                      )

    """ ------- terminate learning """
    print("Saving model to LunarLander_OpenAI_baseline_model_OpenAI_baseline_model.pkl")
    act.save("{}/LunarLander_OpenAI_baseline_model.pkl".format(dir))
