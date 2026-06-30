"""Nanowire candidate detection, topology tiering, and the optional CNN.

The CNN is an optional "second opinion": torch is only required for the model
training/inference entry points and is imported lazily, so the detection
pipeline runs without it (install the ``cnn`` extra to enable the CNN).
"""
