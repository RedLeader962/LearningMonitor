# -*- coding: UTF-8 -*-
from gym.wrappers import monitor
from monitoring.video_recorder.image_sequence_as_array_recorder import ImageSequenceAsArrayRecorder
from monitoring.stats_recorder import LearningStatRecorder

from gym import Wrapper
from gym import error, version, logger
import os, json, numpy as np, six
from gym.wrappers.monitoring import stats_recorder, video_recorder
from gym.utils import atomic_write, closer
from gym.utils.json_utils import json_encode_np


class LearningMonitor(monitor.Monitor):
    def __init__(self, env, directory, learning_monitor_config, plot_training_in_real_time=False, force=False, resume=False, write_upon_reset=False, uid=None, mode=None):
        """
        Wrap a OpenAi monitoring object around a 'env'.
        You must call explicitly the 'get_learning_data_collector' method if you want to monitor the learning algorithm
        or use the alternate constructor.

            ex:
                monitored_env, learning_data_collector =  LearningMonitor.build_collectors(...)

        Note:
            - 'LearningMonitor' turn off realtime env rendering since it's result is push to file
            - To look at rendered run, call the 'TrainingMonitorReader'

        :param env: a OpenAi gym environment
        :param directory:
        :param learning_monitor_config: a config dict
        :param plot_training_in_real_time: plot the training graph to screen in real time
        :param force:
        :param resume:
        :param write_upon_reset:
        :param uid:
        :param mode:
        """
        super(monitor.Monitor, self).__init__(env)

        self.learning_monitor_config = learning_monitor_config
        self.plot_training_in_real_time = plot_training_in_real_time

        self.videos = []

        self.stats_recorder = None
        self.video_recorder = None
        self.enabled = False
        self.episode_id = 0
        self._monitor_id = None
        self.env_semantics_autoreset = env.metadata.get('semantics.autoreset')

        self._start(directory, learning_monitor_config['collect_ep_data_every_callable'], force, resume,
                    write_upon_reset, uid, mode)

    @classmethod
    def build_collectors(cls, env, directory, learning_monitor_config, plot_training_in_real_time=False,
                         force=False,
                         resume=False, write_upon_reset=False, uid=None, mode=None):
        """
        Convenient Alternate constructor that build the monitored env & the learning_data_collector object in one call

        :param env:
        :param directory:
        :param learning_monitor_config:
        :param plot_training_in_real_time:
        :param force:
        :param resume:
        :param write_upon_reset:
        :param uid:
        :param mode:
        :return: the wrapped env & the learning_data_collector object
        """
        monitored_env = cls(env, directory, learning_monitor_config, plot_training_in_real_time, force,
                            resume, write_upon_reset, uid, mode)
        return monitored_env, monitored_env.get_learning_data_collector()

    def get_learning_data_collector(self):
        return self.stats_recorder.create_collector()

    def _start(self, directory, collect_ep_data_every_callable=None, force=False, resume=False, write_upon_reset=False, uid=None, mode=None):
        """Start monitoring.
        
                Args:
                    directory (str): A per-training run directory where to record stats.
                    collect_ep_data_every_callable (Optional[function, False]): function that takes in the index of the episode and outputs a boolean, indicating whether we should record a data on this episode. The default (for collect_ep_data_callable is None) is to take perfect cubes, capped at 1000. False disables data recording.
                    force (bool): Clear out existing training data from this directory (by deleting every file prefixed with "openaigym.").
                    resume (bool): Retain the training data already in this directory, which will be merged with our new data
                    write_upon_reset (bool): Write the manifest file on each reset. (This is currently a JSON file, so writing it is somewhat expensive.)
                    uid (Optional[str]): A unique id used as part of the suffix for the file. By default, uses os.getpid().
                    mode (['evaluation', 'training']): Whether this is an evaluation or training episode.
        """

        if self.env.spec is None:
            logger.warn(
                "Trying to monitor an environment which has no 'spec' set. This usually means you did not create it via 'gym.make', and is recommended only for advanced users.")
            env_id = '(unknown)'
        else:
            env_id = self.env.spec.id

        if not os.path.exists(directory):
            logger.info('Creating monitor directory %s', directory)
            if six.PY3:
                os.makedirs(directory, exist_ok=True)
            else:
                os.makedirs(directory)

        if collect_ep_data_every_callable is None:
            collect_ep_data_every_callable = monitor.capped_cubic_video_schedule
        elif collect_ep_data_every_callable == False:
            collect_ep_data_every_callable = monitor.disable_videos
        elif not callable(collect_ep_data_every_callable):
            raise error.Error('You must provide a function, None, or False for collect_ep_data_every_callable'
                              ', not {}: {}'.format(type(collect_ep_data_every_callable), collect_ep_data_every_callable))
        self.video_callable = collect_ep_data_every_callable

        # Check on whether we need to clear anything
        if force:
            monitor.clear_monitor_files(directory)
        elif not resume:
            training_manifests = monitor.detect_training_manifests(directory)
            if len(training_manifests) > 0:
                raise error.Error('''Trying to write to monitor directory {} with existing monitor files: {}.

        You should use a unique directory for each training run, or use 'force=True' to automatically clear previous monitor files.'''.format(
                    directory, ', '.join(training_manifests[:5])))

        self._monitor_id = monitor.monitor_closer.register(self)

        self.enabled = True
        self.directory = os.path.abspath(directory)

        # We use the 'openai-gym' prefix to determine if a file is ours
        self.file_prefix = monitor.FILE_PREFIX
        self.file_infix = '{}.{}'.format(self._monitor_id, uid if uid else os.getpid())

        # Note: 'self.stats_recorder' is used by the parent class... we can not change is name
        self.stats_recorder = LearningStatRecorder(directory,
                                                   '{}.episode_batch.{}'.format(self.file_prefix,
                                                                                self.file_infix),
                                                   self.learning_monitor_config,
                                                   autoreset=self.env_semantics_autoreset,
                                                   env_id=env_id)

        if not os.path.exists(directory):
            os.mkdir(directory)
        self.write_upon_reset = write_upon_reset

        if mode is not None:
            self._set_mode(mode)


def learning_monitor_builder(env, directory, learning_monitor_config, plot_training_in_real_time=False, force=False,
                             resume=False, write_upon_reset=False, uid=None, mode=None):
    """
    Convenient function for building the monitored env & the learning_data_collector object
    :param env:
    :param directory:
    :param learning_monitor_config:
    :param plot_training_in_real_time:
    :param force:
    :param resume:
    :param write_upon_reset:
    :param uid:
    :param mode:
    :return: the wrapped env & the learning_data_collector object
    """
    monitored_env = LearningMonitor(env, directory, learning_monitor_config, plot_training_in_real_time, force,
                                    resume, write_upon_reset, uid, mode)
    learning_data_collector = monitored_env.get_learning_data_collector()
    return monitored_env, learning_data_collector

