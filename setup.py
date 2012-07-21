from distutils.core import setup, Extension
import numpy as np

interop = Extension('interop',
                    sources = ['interop.c'],
                    include_dirs = ['/hydra/S1/local/MATLAB/R2011a/extern/include',
                                    np.get_include()],
                    library_dirs = ['/hydra/S1/local/MATLAB/R2011a/bin/glnxa64'],
                    libraries = ['eng'])

setup (name = 'matlab',
       version = '1.0',
       description = 'Provides bindings for the MATLAB engine C API',
       ext_modules = [interop])
