from distutils.core import setup, Extension
import numpy as np

interop = Extension('interop',
                    sources = ['interop.c'],
                    include_dirs = ['/usr/local/MATLAB/R2012b/extern/include',
                                    np.get_include()],
                    library_dirs = ['/usr/local/MATLAB/R2012b/bin/glnxa64'],
                    libraries = ['eng'])

setup (name = 'matlab',
       version = '1.0',
       description = 'Provides bindings for the MATLAB engine C API',
       ext_modules = [interop])
