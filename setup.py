#!/usr/bin/env python3

import os
import sys
from setuptools import setup
from setuptools.command.install import install
from setuptools.dist import Distribution


class InstallPlatlib(install):
  """Fix auditwheel error, https://github.com/google/or-tools/issues/616"""

  def finalize_options(self) -> None:
    install.finalize_options(self)
    if self.distribution.has_ext_modules():
      self.install_lib = self.install_platlib


class BinaryDistribution(Distribution):

  def is_pure(self) -> bool:
    return False

  def has_ext_modules(foo) -> bool:
    return True


if __name__ == '__main__':
  # Set PYTHON_EXECUTABLE environment variable so CMake uses the correct Python
  # This ensures the compiled module matches the Python version being used
  if 'PYTHON_EXECUTABLE' not in os.environ:
    os.environ['PYTHON_EXECUTABLE'] = sys.executable
  
  setup(distclass=BinaryDistribution, cmdclass={'install': InstallPlatlib})
