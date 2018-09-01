# -*- coding: UTF-8 -*-
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from monitoring.video_recorder import image_sequence_as_array_writer

class ImageSequenceAsArrayRecorder(VideoRecorder):
    def __init__(self, env, path=None, metadata=None, enabled=True, base_path=None):
        self.unnamed_LearningMonitor_ = None
        # @AssociationType LearningMonitor
        self.unnamed_ImageSequenceAsArrayWriter_ = None
        # @AssociationType monitoring.video_recorder.ImageSequenceAsArrayWriter
        # @AssociationKind Composition

    def _encode_image_frame(self, frame):
        """@ReturnType None"""
        pass

    def _encode_ansi_frame(self, frame):
        """@ReturnType None"""
        pass

