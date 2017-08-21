'''
Created on Aug 18, 2017

Setup file for py2exe

@author: Renat
'''

from distutils.core import setup
import py2exe
import numpy as np
from glob import glob
import sys

data_files = [("Microsoft.VC90.CRT", glob(r'C:\\Program Files (x86)\\Microsoft Visual Studio 9.0\\VC\\redist\\x86\\Microsoft.VC90.CRT\\*.*'))]
sys.path.append("C:\\Program Files (x86)\\Microsoft Visual Studio 9.0\\VC\\redist\\x86\\Microsoft.VC90.CRT")
setup(data_files=data_files, console=['EmbImgLoader.py'],
      options={'py2exe':{"dll_excludes": ["MSVCP90.dll","libzmq.pyd","geos_c.dll",'api-ms-win-core-stringansi-l1-1-0.dll', 'api-ms-win-core-privateprofile-l1-1-1.dll','api-ms-win-core-libraryloader-l1-2-2.dll','api-ms-win-mm-time-l1-1-0.dll','api-ms-win-core-registry-l2-2-0.dll','api-ms-win-core-rtlsupport-l1-2-0.dll','api-ms-win-core-heap-obsolete-l1-1-0.dll','api-ms-win-core-largeinteger-l1-1-0.dll','api-ms-win-core-string-obsolete-l1-1-0.dll', "api-ms-win-core-string-l1-1-0.dll","api-ms-win-core-registry-l1-1-0.dll","api-ms-win-core-errorhandling-l1-1-1.dll","api-ms-win-core-string-l2-1-0.dll","api-ms-win-core-profile-l1-1-0.dll","api-ms-win*.dll","api-ms-win-core-processthreads-l1-1-2.dll","api-ms-win-core-libraryloader-l1-2-1.dll","api-ms-win-core-file-l1-2-1.dll","api-ms-win-security-base-l1-2-0.dll","api-ms-win-eventing-provider-l1-1-0.dll","api-ms-win-core-heap-l2-1-0.dll","api-ms-win-core-libraryloader-l1-2-0.dll","api-ms-win-core-localization-l1-2-1.dll","api-ms-win-core-sysinfo-l1-2-1.dll","api-ms-win-core-synch-l1-2-0.dll","api-ms-win-core-heap-l1-2-0.dll","api-ms-win-core-handle-l1-1-0.dll","api-ms-win-core-io-l1-1-1.dll","api-ms-win-core-com-l1-1-1.dll","api-ms-win-core-memory-l1-1-2.dll","api-ms-win-core-version-l1-1-1.dll","api-ms-win-core-version-l1-1-0.dll"],
               'includes': ['scipy.sparse.csgraph._validation',
                            'scipy', 'scipy.integrate', 'scipy.special.*','scipy.linalg.*']}
               })