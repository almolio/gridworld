# Windows
Use Anaconda to install what you need 

Go into `requirements.yml` file and change the name of the environment if you wish. The current default will be DRL 

Navigate into envsetup folder on your machine with your conda terminal and run: 

`conda env create -f .\requirements.yml`

This will install everything except for torch realated stuff.
Activate the environment you have just created. The default is DRL:

`conda activate DRL`

## Your system install:

We can install the rest with pip. Follow this link https://pytorch.org/ . It will look something like this, but depending on your system requirement 

`pip3 install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118`

and 

`pip3 install pytorch-lightning==2.0.2`


# Linux
Have not yet tested but should be similar to windows and the path slash would be /

### Developer notes: 
conda env export --no-builds > \{filename\}.yml

Try to install with `conda install ...`

