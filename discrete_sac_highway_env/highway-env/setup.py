#import os
from setuptools import setup
from Cython.Build import cythonize

#os.environ['CFLAGS'] = '-O3 -Wall -std=c++11 -ffast-math'

setup(
    ext_modules = cythonize(["highway_env/road/lane.pyx",
                             "highway_env/road/road.pyx",
                             # "highway_env/vehicle/kinematics.py",
                             "highway_env/vehicle/controller.pyx",
                             "highway_env/utils.pyx"],
                            # compiler_directives={'embedsignature': True},
                            annotate=True)
)

