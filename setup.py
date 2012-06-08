from distutils.core import setup, Extension

interop = Extension('interop',
                    sources = ['interop.c'],
										include_dirs = ['/hydra/S1/local/MATLAB/R2011a/extern/include',
																		'/home/kmatzen/homebrew/Cellar/python3/3.2.3/lib/python3.2/site-packages/numpy/core/include'],
										library_dirs = ['/hydra/S1/local/MATLAB/R2011a/bin/glnxa64'],
										libraries = ['eng'])

setup (name = 'matlab',
       version = '1.0',
       description = 'Does nothing more than provide bindings for the MATLAB engine C API',
       ext_modules = [interop])
