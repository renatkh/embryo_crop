### ABOUT embryoCropUI.py, screenCrop.py v1.0 ###

References to the methods used in this application can be found in:

_A high-content imaging method for monitoring C. elegans embryonic development S. Wang1,2†, S.D. Ochoa1,3†, R.N. Khaliullin1,4†, Adina Gerson-Gurwitz1,5, Jeffrey Hendel1, Zhiling Zhao1, Ronald Biggs1, Andrew D. Chisholm6, Arshad Desai1, Karen Oegema1 *, and Rebecca A. Green1 *_

This repository https://github.com/renatkh/embryo_crop consists of two main Python programs:

1.	**embryoCropUI**: graphical user interface (GUI) version of our software, which can accommodate individual image stacks from a range of imaging platforms. This is the source code for a user-friendly executable version, which crops, rotates anterior-posterior, processes for drift correction, background subtraction and attenuation correction for individual image stacks. Cropped images are automatically saved to a “crop” file in the location of the file selected to be cropped. Available as a platform specific executable (stand-alone program for MacOS or Windows that does not require Python installation, which can be found at (https://zenodo.org/record/1475442#.W9jvApNKiUl) or in .py format (this repository).  

2.	**screenCrop.py**: Takes in multiwell, multipoint imaging data and crops, rotates anterior-posterior, processes for drift correction, and performs background subtraction and attenuation correction in batch. Requires Python, virtual environment, and .csv file with the file format and condition specifications delineated. Saves cropped files to new “Cropped” folder with file structure as specified in .csv file. This program is specific to our file structure and acquisition parameters, but can be modified to accommodate similarly structured data if users have Python expertise.

### SETUP ###
Both programs (embryoCropUI.py and screenCrop.py)use specific versions of Python and Python modules, thus configuring an appropriate environment is essential for the programs to run. We recommend and provide instructions for installation of Git, Anaconda (includes Python3), and PyCharm to enable proper environment configuration (detailed instructions below). *Note- MacOS distribution of embryoCropUI.UI is only compatible with Mac OS X 10.11 and higher* 

### DETAILED INSTRUCTIONS ###

1.	**Configure your environment**

    *Clone repository with GIT*

      1. If you don’t already have GIT installed, go to https://git-scm.com/download/ . You may need to enable security settings to be            sure it will download.
      2. Check install by going to terminal or command prompt and enter:
       
           >git --version
       
     * if installed a version will be listed in the terminal* 
      3. Clone repository:
    
           >git clone https://github.com/renatkh/embryo_crop.git
      4. Check in your home directory to ensure that it was properly downloaded.


    *Install Visual Studio (WINDOWS ONLY):*
      1.	Go to www.visualstudio.microsoft.com/downloads and download Visual Studio. This contains C++ tools, which are required for proper setup of the virtual environment with anaconda .yml files.
      2.	Select C++ tools
      3.	Install


    *Setup virtual environment with Anaconda*
      1. If you don’t already have Anaconda, go to www.anaconda.com/download/ and Download Anaconda3 (python3.7 version), launch anaconda setup and click through default options to install.
      2. set environment variables and add conda to the path:
      * Find conda.exe location in __Anaconda Prompt__:
         * WINDOWS: Go to windows button-> Anaconda3-> Anaconda Prompt 
         * MacOS: Anaconda3-> Anaconda Prompt
      * At the prompt type in 
          > where conda
      * Find the location where conda.exe is located (ignore the .bat location) so you can add this location to environmental variables (this can be done within Anaconda Prompt).
      ![whereconda](https://user-images.githubusercontent.com/38474936/47879250-5fd36300-dddd-11e8-8107-ceadd63a0ec2.jpg)
      * In this case it is C:\Users\rebec\Anaconda3\Scripts, but **obviously this will be specific to your system, so please edit the path appropriately!** For this example, we need to add both paths:
          *  C:\Users\rebec\Anaconda3
          *  C:\Users\rebec\Anaconda3\Scripts
      * *Add to environment variables*. To do this, type:
          *  >SETX PATH "%PATH%;C:\Users\rebec\Anaconda3;C:\Users\rebec\Anaconda3\Scripts”
	  ![path](https://user-images.githubusercontent.com/38474936/47886213-06c3f900-ddf6-11e8-85f7-dcf027fe5be0.jpg)
      3. Close Anaconda Prompt
      4. Go to system terminal or command prompt (not Anaconda prompt) and check to be sure that the conda command works.
	        *  >conda
          * this should return information about conda functionality. If it does not, you have not successfully added path environmental variables.
      5. Configure the environment in command line/terminal:
      * Navigate to the location where embryo_crop repository was saved. 
      * In our example, it is saved here: C:\Users\rebec\embryo_crop, so at the prompt: 
           >cd C:\Users\rebec\embryo_crop

      * Create new conda environment from .yml file
           * 1. Once inside the directory, create the environment:
             * -for Windows: 
               * > conda env create -f environment_win.yml
             * -for MacOS:
               * > conda env create -f environment_mac.yml

           *  This step will take a few minutes to solve the environment...
	   ![solvingenv](https://user-images.githubusercontent.com/38474936/47886420-2c9dcd80-ddf7-11e8-8a38-dcf83ae508fe.jpg)
           
     6. When finished, you can continue in command line to run embryoCropUI (below) or switch to an IDE to run screenCrop or embryoCropUI.
	   * *To continue in command line (for embryoCropUI)*:
	       * Activate the environment according to the instructions listed in the terminal. 
              *  -for Windows: > activate embryocrop
              *  -for MacOS:: > source activate embryocrop
     * Once activated, you can run python programs by calling the program in command line
          * >python embryoCropUI.py
          * this will launch the GUI window- please follow the instructions for the GUI use (in the readme file).
     * When finished:
          * Close GUI  
          * >deactivate

 1. **Configure Environment in IDE (Setup in PyCharm)** 
  * To access the code directly, which is necessary for screenCrop.py functionality, open the environment in your favorite Integrated Development Environment (IDE). We include instructions for installing and running with PyCharm. Note that Jupyter notebooks currently DOES NOT support GUIs, so our embryoCropUI will not run properly in this environment. To get PyCharm, go to: https://www.jetbrains.com/pycharm/download/#section=windows. Select your operating system and click the black download button under “Community”.

  * In **PyCharm**: 
    1.	**File > Open > embryo_crop** 
    2.	*Configure environment*. Go to: **file > settings > project:embryo_crop > project interpreter > add local (select conda) > existing environment >** and select the newly generated conda env from within the Anaconda3 envs folder: **…Anaconda3\envs\embryocrop\python.exe**
    ![pycharmenv](https://user-images.githubusercontent.com/38474936/47886427-3b848000-ddf7-11e8-9c16-942f3536d664.jpg)
    3.	From here, you should be able to run programs using the PyCharm ‘run’ button (program instructions below). If this doesn’t properly structure the environment, it may crash. If this happens, you can access the terminal window within PyCharm and activate the environment this way:
      * a.	For Windows:
          * > activate embryocrop
      * b.	For MacOS:
         * > source activate embryocrop
      * c. Once the environment is activated, you can run the program via the terminal within PyCharm:
         * > python screenCrop.py  or 
         * > python embryoCropUI.py
      * *note that the screenCrop program will need to be modified to work with your file structure!!See instructions below.*
      ![pycharmterminal](https://user-images.githubusercontent.com/38474936/47886459-7090d280-ddf7-11e8-99c3-fb499942af2c.jpg)
      * d. When finished:
         * > deactivate

2. **Running embryoCropUI.py GUI with Python**
    1.	In PyCharm (or your preferred IDE) double click on the embryoCrop folder and locate the file that says embryoCropUI.py  (DO NOT open embryoCropUI.ui). The code will appear in the workspace. 
    2.	If this is the only file open, go to the top right-hand corner and click the green triangle to start the run. If multiple files are open, right click and select ‘Run embryoCropUI’ to ensure the proper program is run. Alternatively, activate the environment and run from PyCharm terminal, as outlined above.
    3.	Once the program has started, the following window will appear: 
      * ![gui1](https://user-images.githubusercontent.com/38474936/47886470-81414880-ddf7-11e8-94f7-cef002f8f564.jpg)
    4.	Select the “Open” button at the top of the window to load the specific image that you wish to crop. Should you be cropping an image series, with multiple dimensions (i.e. z, time, or channel), simply load the first image in the series within the folder. *Please make sure only images from one image series are present in this folder, otherwise the image series’ will be loaded in tandem.*
    5.	Once you have loaded the desired images, you will need to specify the following information:
      a.	Number of Z slices (Z)
      b.	Number of Time points (T)
      c.	Number of Channels (C) 
      d.	The channel that corresponds to DIC or brightfield (first=1, second=2, etc)
     * For example: our imaging protocol was 18 z-steps, imaged for 31 time-points in 3 channels with the DIC channel being the first channel imaged. The window should look like this: 
     ![gui2](https://user-images.githubusercontent.com/38474936/47886476-8d2d0a80-ddf7-11e8-8e0c-f920ebc05de8.jpg)
    6.	Now that you have your images loaded and specified the image parameters, you must choose what processing you would like to do alongside the embryo cropping. The program gives you the option to perform Drift Correction, Background Subtraction, and Attenuation Correction. Background Subtraction and Attenuation Correction must be done in conjunction with each other. The below image shows an image that will be going through Drift Correction, Background Subtraction, and Attenuation Correction. 
      * ![gui3](https://user-images.githubusercontent.com/38474936/47886489-a0d87100-ddf7-11e8-9e01-3e951a24dcb2.jpg)
      a.	When selecting Background Subtraction and Attenuation Correction, specify parameters for each to guide the processing. For Background Subtract, define a feature size (odd numbered) that reflects the level of detail you wish to resolve, larger feature size equates to more detail. For Attenuation Correction, you need to input a value from 0-1. This value represents the percent of original intensity that remains at the furthest distance through the object being imaged.
      b.	As shown below, you have even greater options to customize Background subtraction. By selecting Customize, you will be able to define a feature size for different channels.
      * ![gui4](https://user-images.githubusercontent.com/38474936/47886497-adf56000-ddf7-11e8-909f-de7eae36df89.jpg)
    7. Next, specify the order in which the images were collected (i.e. channel-z-time (czt), or z-channel-time (zct))
    8. Specify the microns per pixel of the camera being used for the images.
      a. failure to properly define pixel size will result in poor image cropping!!
    9.	Select Run at the bottom left corner and the program. When the cropping and processing of your images has completed, the cropped versions will be saved in a new subfolder labeled “crop” in the same folder as the uncropped images.

  * **We make available two formats of test files for testing embryoCropUI.py. This folder is too large for Github requirements and thus is stored on the Zenodo repository (https://zenodo.org/record/1475442#.W9jvApNKiUl)**. Download and unzip. We recommend testing one or both to ensure the program is functioning properly on your system:
    i.	TESTME2_BGLI140_1t_a1.tif- a compiled multi-tif format
      1.	Load file in the ‘open’ field. Set Z=1, T=6, C=3, DIC=1 and use the default settings for all other fields. Click Run. If successful, a message will appear at the bottom of the GUI window that says ‘embryos saved’ and it will generate a folder in the same location as the test file labeled “crop”; this should contain 4 embryos.
    ii.	Test_field- a folder containing an short image series
      1.	Load the first image in the test_field folder into the ‘open’ window. Set Z=18, T=4, C=3, DIC=3, change the pixel size to 0.26um/pix. Click Run. a message will appear at the bottom of the GUI window that says ‘embryos saved’ and it will generate a folder in the same location as the test file labeled “crop”; this should contain 2 embryos.
* *These files should crop in seconds to minutes, but larger image sequences may take some time. The bottom corner of the GUI window will read-out what the program is doing (“Loading images”, “cropping”, or “embryos saved”). If an error occurs, the message will appear here.*

3.**Running screenCrop.py**
  * This software allows you to batch crop many files at once, but it is less user friendly and has not been optimized across platforms. It was designed to function with data output from CV1000 imaging systems. If you have another system, modifications to the code will likely be required and someone with Python experience will be needed. In the event that it is needed, we outline key elements of the code and our file structure to guide such efforts. Successful bulk cropping requires:
    1.	A reference .csv file that contains essential image information, which is called by our Python software
    2.	Properly named files
    
      * **CSV**: The CSV file contains information that will be called on during processing or used to generate the cropped file path. Below is an example .csv file that is compatible with our programs. Formatting your .csv file the same will ensure your data will go through our programs with minimal issues.
        * Rundown of .CSV file contents:
	![csv](https://user-images.githubusercontent.com/38474936/47886502-b8aff500-ddf7-11e8-9bf4-1451b68fb1d4.jpg)
          * Experiment: arbitrary name given to each experiment (not important for software functionality, but this column needs to be maintained)
          * Experiment Folder Name: name of folder in where specific experiments images are stored. We prefer Date/Time file name, though any name will suffice (do not include spaces or disallowed characters, as the contents of this cell are added to the file path).
          * Post-Scan Folder Name: name of post- scan (10x data) folder. Not important for software functionality, but this column needs to be maintained. You can populate this with ‘empty’. 
          * Well Designation: Well numbers as determined by CV1000 software adhering to the Well### regime.
          * Target: Experimental conditions (e.g. RNAi condition), we use a blinded, unique identifier system (EMBD####) for our experimental conditions, though this is not necessary. Output files will be saved according to this name, so do not include spaces or disallowed characters.
          * Strain: Specific strain used in experiment. Scaling and background subtraction is applied differently depending on the strain used.
          * Plate Coordinate: Coordinates from 384 well plate (for reference, not used by program).

      * **Properly naming files**:
          * Input path:
            * The path to access the raw data files is referred to in screenCrop.py, based on the information in the CSV file (highlighted in red below). Our data is structured such that multiple point visits are contained within each well folder and all of the image files are listed within that well (not in separate subfolders per point visit). Image files have been automatically named according to CV1000 software image naming conventions, as follows:
            * Z:\CV1000\Experiment Folder Name\Well designation\W##F##T###Z##C#.tif
            * i.e. Z:\CV1000\20180809T122430\Well001\W1F001T0001Z01C1.tif.
  * ![filestructure](https://user-images.githubusercontent.com/38474936/47886512-c49bb700-ddf7-11e8-8679-d09df3c1ec2c.jpg)
  	  * Output path:
            * The output path specified in screenCrop.py is also generated based on information in the CSV file (highlighted in red below). Our saving regime uses our naming scheme (Emb#) for outer folders representing data for each embryo and this contains individual tifs that are named as follows:
            * Z:\cropped\Target\Strain\Emb#\Target_Emb#_Experiment Folder Name_W##F#_T##_Z##_C#.tif
            * i.e. Z:\cropped\EMBD0002\MS\Emb1\EMBD0002_Emb1_20140402T140154_W06F2_T01_Z01_C1
  * ![filestructure_out](https://user-images.githubusercontent.com/38474936/47886524-d5e4c380-ddf7-11e8-9063-ed62afda694f.jpg)

  3. Cropping your images using **ScreenCrop.py**-- you will be able to crop all your images from a folder. The program crops each image by fitting an ellipse to each embryo at the fourth time point. 
      * Open PyCharm, or other IDE, load the embryo_crop repository and locate the program screenCrop.py. It is important to know the following Information and fill it in at the specified line:
        * 	loadFolder (line 9): The drive on which the files are stored (e.g. Z:/ , D:// etc.)
        * 	date (line 7): this is the file referred to as Experiment Folder Name in the CSV 
        * 	trackingFile (line 11): the path to the CSV file in which experiment information is stored
        * 	z (line 13): The number of z planes
        * 	nT (line14): Number of timepoints
        * 	nWells (line 19): the number of wells used
        * 	pointVisits (line 20): the maximum number of point visits (per well)
        * 	In line 10 find the location currently occupied by ‘CV1000/’ and input the outer folder used in your file path. To avoid issues, use the following convention: ‘XXXXXXX/’. 
        * 	In Line 12, input a valid file path for storing aspect ratio data for the cropped.
        * 	In Lines 15, 16, 17, and 18 input True/False for whether you would like your images to go through the following processing:
        * 	Drift Correction (Line 15)
        * 	Background Subtract (Line 16), feature size should be defined as 41 for GLS strain and 201 for MS strain. Background Subtract must be done in conjunction with Attenuation Correction. 
        * 	Attenuation Correction (Line 17)
        * 	AP Rotate (Line 18)
      * Once all the changes have been made to tailor the program for your data, you may begin cropping. This is done by selecting the green play icon in the toolbar, this will have a drop down menu where you select “Run As” and then “Python Run”. Alternatively, activate the environment and run from PyCharm terminal (described above at the end of the PyCharm section).
      * The Program will then begin cropping your images, this may take a few hours depending on the number of images that need to be processed. Once completed, a series of small windows containing embryo images will open; this will allow you to curate the cropped data before saving (i.e. delete embryos that are cut off, out of focus, or poorly cropped can be deleted). 
      * Good! 
        * ![good](https://user-images.githubusercontent.com/38474936/47886532-e4cb7600-ddf7-11e8-8abf-df3087729a23.jpg)
      * Bad!! 
        * ![bad](https://user-images.githubusercontent.com/38474936/47886540-f3b22880-ddf7-11e8-81de-9ed1e902eed2.jpg)
      * For each image you have three options: 
        *  Save: If the image appears to be cropped properly with no areas of interest being cut off, press the space bar to save the image. 
        *  X: If the image appears to have areas of interest cut off and you still wish to save the image, press X and the image will be saved with an X in front of the name to separate it from the others. 
        *  Delete: If the image is not cropped properly or the embryo is not to your liking, press D to delete the cropped image. 
      * Once you have gone through all your images and determined whether you wish to save, x, or delete them, the program will then begin to save your images. The images will be saved to a subfolder named “Cropped” in the Load Folder that was defined in Line 9 of the program.



### CONTACT ###
* Renat Khaliullin renatkh@gmail.com
* Rebecca Green regreen@ucsd.edu


