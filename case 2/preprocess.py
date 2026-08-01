"""Reshape a recording before the model or agent sees it, and undo it after.

Applied at two boundaries:

    distill : transform_distill  before the model trains/predicts
              revert_distill      to bring predictions back to real units
    rla     : transform_rla       before the agent observes the dataset
              revert_rla          to bring agent-produced values back

``Preprocess`` is the interface. Each method takes a recording DataFrame and
returns one. ``transform_*`` maps it into the space the model or agent works in;
``revert_*`` must be the exact inverse, so a value that makes a round trip comes
back unchanged.

The default ``Identity`` returns the frame unchanged, so the pipeline runs end to
end with no preprocessing. The Preprocess object is passed explicitly into the
distill trainer and the RLA, so training and prediction use the same one.

Examples of what a subclass might do:

- Normalize columns on very different scales (radians vs raw movej numbers up to
  ~1000); ``revert_*`` undoes the scaling on the predicted channels.
- Log- or arcsinh-scale the current so large swings do not dominate.
- Difference or detrend a channel, then integrate back in ``revert_*``.

    from preprocess import Preprocess
    import numpy as np

    class ScaleCurrent(Preprocess):
        \"\"\"Divide every current column by 10 for the distill model, undo after.\"\"\"
        def transform_distill(self, df):
            df = df.copy()
            cols = [c for c in df.columns if "current" in c]
            df[cols] /= 10.0
            return df
        def revert_distill(self, df):
            df = df.copy()
            cols = [c for c in df.columns if "current" in c]
            df[cols] *= 10.0
            return df
        def transform_rla(self, df):
            return df
        def revert_rla(self, df):
            return df
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Preprocess(ABC):
    """Interface for reshaping a recording into and out of a learner's space.

    Every method takes a recording DataFrame and returns a DataFrame. Implement
    all four; keep each ``revert_*`` the exact inverse of its ``transform_*``.
    """

    @abstractmethod
    def transform_distill(self, df):
        """Map a recording into the space the distill model trains/predicts in."""

    @abstractmethod
    def revert_distill(self, df):
        """Inverse of ``transform_distill``: back to real units after predicting."""

    @abstractmethod
    def transform_rla(self, df):
        """Map a recording into the space the RL agent observes."""

    @abstractmethod
    def revert_rla(self, df):
        """Inverse of ``transform_rla``: back to real units for agent outputs."""


class Identity(Preprocess):
    """Do-nothing default: every method returns the frame unchanged."""

    def transform_distill(self, df):
        return df

    def revert_distill(self, df):
        return df

    def transform_rla(self, df):
        return df

    def revert_rla(self, df):
        return df


def default_preprocess() -> Preprocess:
    """The Preprocess the whole pipeline uses.

    Distill training, RLA training, and prediction (run.py) all get their
    Preprocess from here, so they stay consistent. Return a Preprocess subclass
    here to apply it to every stage.
    """
    return Identity()
