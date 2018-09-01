# -*- coding: UTF-8 -*-
class TrainingMonitorReader(object):
    def set_extra_plotting_callback(self, callback):
        pass

    def _render_image_sequence(self, directory):
        pass

    def _render_plot_sequence(self, directory):
        pass

    def render_episode_sequence(self, episode_intervale, rendering_size="HD"):
        pass

    def read_all_rendered(self):
        pass

    def read_episodes(self, from_6 = None, to = None, keep = True):
        pass

    def wipe_all_rendered(self):
        pass

    def wipe_rendered_episode(self, from_7 = None, to = None):
        pass

    def get_rendered_files_size(self):
        """@ReturnType float"""
        pass

    def __init__(self):
        self.plotting_callback_registry = []

