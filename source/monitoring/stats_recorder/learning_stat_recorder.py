# -*- coding: UTF-8 -*-
import json
import os
import time

from gym.wrappers.monitoring.stats_recorder import StatsRecorder
from monitoring.stats_recorder.learning_data_collector import LearningDataCollector

from gym.utils import atomic_write
from gym.utils.json_utils import json_encode_np


class LearningStatRecorder(StatsRecorder):
    def __init__(self, directory, file_prefix, learning_monitor_config, autoreset=False, env_id=None):

        super(LearningStatRecorder, self).__init__(directory, file_prefix, autoreset, env_id)

        self.episode_idx = 0
        self.exploration_rates = []
        self.mean_loss = []
        self.episode_start_keyframe = []
        self.event_keyframe_registry = []
        self.episodes_dataClass = []

    def after_step(self, observation, reward, done, info):
        # TODO
        # pass

        # Note: -- temp ---------------------------
        self.steps += 1
        self.total_steps += 1
        self.rewards += reward
        self.done = done

        if done:
            self.save_complete()

        if done:
            if self.autoreset:
                self.before_reset()
                self.after_reset(observation)



    def after_reset(self, observation):
        # TODO
        # pass

        # Note: -- temp ---------------------------
        self.steps = 0
        self.rewards = 0
        # We write the type at the beginning of the episode. If a user
        # changes the type, it's more natural for it to apply next
        # time the user calls reset().
        self.episode_types.append(self._type)

    def save_complete(self):
        # TODO
        # pass

        # Note: -- temp ---------------------------
        if self.steps is not None:
            self.episode_lengths.append(self.steps)
            self.episode_rewards.append(float(self.rewards))
            self.timestamps.append(time.time())

    def flush(self):
        # TODO
        # pass

        # Note: -- temp ---------------------------
        if self.closed:
            return

        with atomic_write.atomic_write(self.path) as f:
            json.dump({
                'initial_reset_timestamp': self.initial_reset_timestamp,
                'timestamps': self.timestamps,
                'episode_lengths': self.episode_lengths,
                'episode_rewards': self.episode_rewards,
                'episode_types': self.episode_types,
            }, f, default=json_encode_np)



    def flush_episodes_dataClass(self):
        # TODO
        pass


    def create_collector(self):
        """Create a LearningDataCollector instance"""
        # TODO
        return LearningDataCollector()

