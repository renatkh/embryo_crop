''' Define adjustable parameters within this file for screenCrop.py. Note that use_config must be changed to True
for screenCrop.py to read these parameters.'''

use_config = False # change this to True to run screenCrop using this parameter configuration file
# (otherwise directly edit the header in screenCrop.py)

date = '20180808T122436'  # unique identifier for experimental condition

loadFolder = 'Z:/'  # parent directory
folderIn = loadFolder + 'CV1000/' + date  # Input folder
trackingFile = 'Z:/Experiment_tracking_sheets/EMBD_fileNames_Tracking_Sheet.csv'  # CSV file to locate conditions to be cropped
aspectRatioFile = 'Z:/cropped/aspects.csv'  # output file to write aspect ratio
z = 18  # number of z planes
nT = 31  # number of time points
corrDrift = True  # drift correction
removeBG = True # background subtraction
attCorrect = True # attenuation correction
apRotate = True # ap rotation
nWells = 1  # number of wells (14)
pointVisits = 1  # number of point visits (4)
