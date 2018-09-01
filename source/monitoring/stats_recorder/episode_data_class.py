# -*- coding: UTF-8 -*-
from monitoring.stats_recorder import learning_data_collector

class EpisodeDataClass(object):
    def __init__(self):
        self.cumulated_reward_per_step = []
        self.reward_delta_per_step = []
        self.observations = []
        self.actions = []


    def __setitem__(self, step_idx, values):
        pass

    def __getitem__(self, step_idx):
        """@ReturnType cumulated_reward_interval, reward_delta_interval, observation, action"""
        pass

