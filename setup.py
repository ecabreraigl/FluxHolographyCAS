#!/usr/bin/env python3
"""
Backward compatibility setup.py for older pip versions.

Modern installations should use pyproject.toml directly.
This file ensures compatibility with pip < 21.3.
"""

from setuptools import setup

# All configuration is in pyproject.toml
# This setup.py is just for backward compatibility
setup()
