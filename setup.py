from setuptools import setup

setup(
  name='optfuncs',
  version='0.0.1',
  description='Optimization functions',
  packages=['optfuncs'],
  install_requires=[
    'tensorflow>=2.7.0',
    'numpy>=1.21.0',
  ]
)