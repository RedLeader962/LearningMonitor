# -*- coding: UTF-8 -*-
from monitoring.stats_recorder import learning_stat_recorder
from monitoring.stats_recorder import episode_data_class


class LearningDataCollector(object):
    def __init__(self, variable_fetch_callback=None, variable_postprocessing_callback=None):
        self.unnamed_EpisodeDataClass_ = []
        # @AssociationType monitoring.stats_recorder.EpisodeDataClass[]
        # @AssociationMultiplicity *
        # @AssociationKind Aggregation

    def __call__(self, lcl, _glb):
        pass

    def teardown(self):
        pass

    def setup_episode_collection(self):
        pass

    def close_episode_collection(self):
        """@ReturnType monitoring.stats_recorder.EpisodeDataClass"""
        pass

