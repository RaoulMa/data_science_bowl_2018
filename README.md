# Data Science Bowl 2018
My code, see https://github.com/RaoulMa/Data_Science_Bowl_2018 or https://www.kaggle.com/raoulma/nuclei-dsb-2018-tensorflow-u-net-score-0-352, implements a U-Net shaped convolutional neural network in TensorFlow for nuclei segmentation of the images provided by the 2018 Data Science Bowl (https://www.kaggle.com/c/data-science-bowl-2018/data).  

## Abstract
The 2018 Data Science Bowl "Find the nuclei in divergent images to advance medical discovery" provides in its first stage a training and test data set consisting of 670 and 65 microscopic images of varying size showing ensembles of cells and their nuclei. For the training images the nuclei are segmented by humans such that we know their number and location within each image. The goal is to find the correct number and location of all nuclei shown in the test images. The performance of an algorithm is evaluated on the mean average precision at different intersection over union (IoU) thresholds, which will be referred to as the score in the following.

For the task we implement a deep neural network of the U-Net type consisting of several convolutional and max-pooling layers. The network is written in TensorFlow. Each model can be saved/loaded and the training process can be visualized with TensorBoard. The input of the network are images of shape (height, width, channels) while the output are corresponding binary masks of shape (height, width, 1).

Prior to training the network we resize, normalize and transform the images. We use 10% of the training data for validation. Furthermore, we implement data augmentation by making use of translations, rotations, horizontal/vertical flipping and zoom. Choosing an image size of 384x384 pixels the network requires roughly 30 training epochs before the training seems to converge. The network achieves a score of 0.56/0.352 on the validation/test set. A major reason for the score discrepancy can be explained by overlapping/touching nuclei that are identified as a single nucleous by the current implementation. Furthermore, we have not tuned the hyperparameters, so there is still a lot of room for improvement.

## Run the Code

The code is tested on
- macOS High Sierra 10.13.4 with Python 3.6.4 Anaconda, TensorFlow 1.4.1 (CPU Computing)
- Ubuntu 16.04.4 LTS with python 3.6.5 Anaconda, GPU Tensorflow 1.8.0-dev,  CUDA 9.1, cuDNN 7.1.2 (GPU Computing)

Models are saved in the folder 'saves', logs for Tensorboard are saved in the folder 'logs' and the image data is contained in the folder 'data'. To execute the code the following files are relevant:
- "sample.sh" is a bash script, which trains a model called 'tmp' and makes predictions including the creation of the submission file. You can modify the file for your purposes. 
- "visualisation.ipynb" is a jupyter notebook which can also be used for training the model. In addition it includes code for the analysis and visualization of the problem. 
- "config.py" contains global variables like the file paths to the image data. You can modify the file for your purposes. 

Run the python script "check_tensorflow.py" to check if tensorflow works properly and if it runs on a CPU or GPU.

## Manual for Google Cloud Computing with a GPU
Manual for setting up a Google Cloud VM instance with a NVidia K80 GPU for deep learning tasks. The virtual machine will be set up with:
- Ubuntu 16.04.4 LTS (GNU/Linux 4.13.0-1012-gcp x86_64) as the OS
- Python 3.6.5 Anaconda
- GPU TensorFlow 1.8.0-dev
- CUDA 9.1
- cuDNN 7.1.2

### Setup Project and VM Instance

Create a google gloud project and upgrade to a paid account if you have not already done it. Add a NVidia K80 GPU to your quota. Then install the gcloud sdk in order to work with the local console. On the gcloud website create a VM instance for the project with 4 Intel Broadwell vCPUs and one NVidia K80 GPU. In order to allow using jupyter with remote access activate for the instance http traffic, https traffic, add jupyter as a tag, and reserve a static ip address for the instance. Have a look at "https://medium.com/@jamsawamsa/running-a-google-cloud-gpu-for-fast-ai-for-free-5f89c707bae6" which is a great reference. 

To get a list of available disk types, run
- $ gcloud compute disk-types list

To get a list of available machine types, run
- $ gcloud compute machine-types list

To list available CPU platforms in given zone, run
- $ gcloud compute zones describe ZONE --format="value(availableCpuPlatforms)"

A list of zones can be fetched by running:
- $ gcloud compute zones list

change default zone of the project
- $ gcloud compute project-info add-metadata --metadata google-compute-default-region=europe-west1,google-compute-default-zone=europe-west1-d
- $ gcloud init

### Log into and Check the VM Instance

log into your vm
- $ gcloud compute ssh user-name@instance-name

check available memory
- $ df -h

look at and set the language locale
- $ locale
- $ sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
- $ export LC_ALL="en_US.UTF-8"

ensure system is updated and has basic build tools
- $ sudo apt-get update
- $ sudo apt-get --assume-yes upgrade
- $ sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils
- $ sudo apt-get --assume-yes install software-properties-common

display GPU
- $ lshw -numeric -C display

verify if GPU is cuda compatible
- $ lspci | grep -i nvidia

check if ubuntu version is supported by cuda
- $ uname -m && cat /etc/*release

### Install Cuda

Have a look at "http://www.python36.com/install-tensorflow-using-official-pip-pacakage/" which is a great reference.

install kernel headers
- $ sudo apt-get install linux-headers-$(uname -r)

install dependencies
- $ sudo apt-get install build-essential
- $ sudo apt-get install cmake git unzip zip
- $ sudo apt-get install python2.7-dev python3.5-dev python3.6-dev pylint

to install linux header supported by your linux kernel type the following command
- $ sudo apt-get install linux-headers-$(uname -r)

download cuda, deinstall old cuda, install downloaded cuda
- $ wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
- $ sudo apt-get purge nvidia*
- $ sudo apt-get auto-remove
- $ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
- $ sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
- $ sudo apt-get update
- $ sudo apt-get install cuda-9.0

reboot system to load the NVIDIA drivers 
- $ sudo reboot

add paths to ~/.bashrc:
- export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
- export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
- $ vim ~/.bashrc
- $ source ~/.bashrc
- $ sudo ldconfig
- $ nvidia-smi

### Install cuDNN

download the following libraries:
- cuDNN v7.1.2 Runtime Library for Ubuntu16.04 (Deb)
- cuDNN v7.1.2 Developer Library for Ubuntu16.04 (Deb)
- cuDNN v7.1.2 Code Samples and User Guide for Ubuntu16.04 (Deb)

- $ wget https://github.com/RaoulMa/Data_Science_Bowl_2018/libcudnn/libcudnn7_7.1.2.21-1+cuda9.1_amd64.deb
- $ wget https://github.com/RaoulMa/Data_Science_Bowl_2018/libcudnn/libcudnn7-dev_7.1.2.21-1+cuda9.1_amd64.deb
- $ wget https://github.com/RaoulMa/Data_Science_Bowl_2018/libcudnn/libcudnn7-doc_7.1.2.21-1+cuda9.1_amd64.deb

install the libraries
- $ sudo dpkg -i libcudnn7_7.1.2.21-1+cuda9.1_amd64.deb
- $ sudo dpkg -i libcudnn7-dev_7.1.2.21-1+cuda9.1_amd64.deb
- $ sudo dpkg -i libcudnn7-doc_7.1.2.21-1+cuda9.1_amd64.deb

Verifying cuDNN installation:
- $ cp -r /usr/src/cudnn_samples_v7/ $HOME
- $ cd  $HOME/cudnn_samples_v7/mnistCUDNN
- $ make clean && make
- $ ./mnistCUDNN

Install dependencies
- $ sudo apt-get install libcupti-dev
- $ echo 'export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
- $ source ~/.bashrc
- $ sudo ldconfig

### Install Python and some Packages

install anaconda
- $ wget http://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh
- $ bash ~/anaconda.sh -p $HOME/anaconda
- $ export PATH="$HOME/anaconda/bin:$PATH"

install virtual environments for python 3.6
- $ conda create -n py36 python=3.6 anaconda

activate or deactivate the environment
- $ source activate py36
- $ source deactivate

update conda, install keras and tqdm packages
- $ conda update -n base conda
- $ conda install -y keras
- $ conda install -y tqdm
- $ conda update --all

install opencv via pip
- $ easy_install -U pip
- $ pip install opencv-python
- $ conda install -y opencv

### Install TensorFlow

install tensorflow for python 2.7 via pip
- $ sudo apt-get install python-pip
- $ easy_install -U pip
- $ pip install --upgrade tensorflow

install gpu tensorflow for python 2.7 via pip
- $ sudo apt-get install python-pip
- $ easy_install -U pip
- $ pip install --upgrade tensorflow-gpu

install tensorflow for python 3.n via pip
- $ sudo apt-get install python3-pip
- $ easy_install -U pip
- $ pip3 install --upgrade tensorflow

install gpu tensorflow for python 3.n via pip
- $ sudo apt-get install python-pip
- $ easy_install -U pip
- $ pip3 install --upgrade tensorflow-gpu

check tensorflow
- $ python -c "import tensorflow; print(tensorflow.__version__)"

in case gpu-tensorflow gives an error
- $ pip install tf-nightly-gpu

### Configure Jupyter

create jupiter config file
- $ jupyter notebook --generate-config

configure jupyter for remote access by creating a firewall rule by using the gc website or by typing the following commands into the console:
- $ export PROJECT="project-name"
- $ export YOUR_IP="instance-ip-address"
- $ gcloud compute --project "${PROJECT}" firewall-rules create "jupyter" --allow tcp:8888 --direction "INGRESS" --
- $ priority "1000" --network "default" --source-ranges "${YOUR_IP}" --target-tags "jupyter"

start jupyter remotely, and use locally the instance IP with the token as password for login
- $ jupyter notebook --ip=0.0.0.0 --port=8888

Open in local browser the url "http://instance-ip-address:8888" and use the token from the console as the password tp log into  jupyter.

### 2018 Data Science Bowl Project

clone the code
- $ git clone https://github.com/RaoulMa/Data_Science_Bowl_2018.git

activate virtual environment
- $ source activate py36

run the code
- $ python sample.sh






