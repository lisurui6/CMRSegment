from pathlib import Path

import numpy as np
import tensorflow as tf
from typing import List

from CMRSegment.subject import Subject, Image, Segmentation


class TF1Segmentor:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        tf.compat.v1.reset_default_graph()
        self._sess = tf.compat.v1.Session()

    def __enter__(self):
        self._sess.__enter__()
        self._sess.run(tf.compat.v1.global_variables_initializer())
        # Import the computation graph and restore the variable values
        saver = tf.compat.v1.train.import_meta_graph('{0}.meta'.format(self.model_path))
        saver.restore(self._sess, '{0}'.format(self.model_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sess.close()
        self._sess.__exit__(exc_type, exc_val, exc_tb)

    def run(self, image: np.ndarray, training: bool = False) -> np.ndarray:
        """Call sess.run()"""
        raise NotImplementedError("Must be implemented by subclasses.")

    def apply(self, image: Image) -> Segmentation:
        """Segment the subject (multiple phases)"""
        np_image, predicted = self.execute(image.path, image.segmented)
        return Segmentation(phase=image.phase, path=image.segmented, image=np_image, predicted=predicted)

    def execute(self, phase_path: Path, output_dir: Path):
        """Segment a 3D volume cardiac phase from phase_path, save to output_dir"""
        raise NotImplementedError("Must be implemented by subclasses.")