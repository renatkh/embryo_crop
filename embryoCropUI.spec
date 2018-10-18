# -*- mode: python -*-

block_cipher = None


a = Analysis(['embryoCropUI.py'],
             pathex=['/home/renat/Documents/work/python/embryocrop'],
             binaries=[],
             datas=[],
             hiddenimports=['scipy._lib.messagestream',
                            'tifffile', 'tifffile._tifffile', '_sysconfigdata', 'cv2',
			    'pywt._extensions._cwt', 'PyQt5', 'PyQt5.QtWidgets', 'PyQt5.QtCore'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='embryoCropUI',
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='embryoCropUI')
#  
