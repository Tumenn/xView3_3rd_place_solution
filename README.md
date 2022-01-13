# xView3_3rd_place_solution
3rd place solution for xView3 challenge https://iuu.xview.us/

For a detailed description of the method, please refer to "./doc/xView3_competition_third_place_solution.pdf".

The submission docker image com be download from docker hub.
```
docker pull sanxia04/heatmap:v4.1
```
## Getting Started

Clone the repository:
```bash
git clone https://github.com/Tumenn/xView3_3rd_place_solution.git
```

We use Python 3.6 and PyTorch 1.9.0 in our implementation, please install dependencies:
```bash
conda create -n xview python=3.6
conda activate xview
conda install pytorch=1.9.0 cudatoolkit=11.1 -c pytorch
conda install GDAL==3.0.2
pip install -r requirements.txt
```

## Data Preparation
First, please download the xview3 competition dataset and unzip it in the folder ./data/raw_data/ .

The structure of the data folder should be:
```
├── data
    ├── raw_data                   # xview3 competition dataset
        ├── train                  # train dataset
        |   ├── 00a035722196ee86t
        |   |   ├── VH_dB.tif
        |   |   ├── VV_dB.tif
        |   |── 0aba50e606eddffet
        |   |── ...
        ├── valid                  # valid dataset
        |   ├── 0d8ed29b0760dc59v
        |   |── ...
        ├── test                   # test dataset
        |
        ├── train.csv              # train label
        ├── valid.csv              # valid label
```

Run the following command to process the xView3 competition dataset :
```bash
cd data_preprocess
python convert_8bit,py
python detection.py
python classfication.py
```

After process, the structure of the data folder should be:
```
├── data
    ├── raw_data                   # xview3 competition dataset
    ├── split_txt                  # store k fold image path
    ├── train_8bit                 # train images in 8 bit
    ├── train_class_crop           # train crops for classification model
    ├── valid_8bit                 # valid images in 8 bit
    ├── valid_class_crop           # valid crops for classification model
    ├── valid_detection_crop       # valid crops for detection model
```

## Training
We use the RTX 3090 GPU to train the network.

Download the pretrained model of [HRNet](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk) and place it under the './data/ckpt/' folder.

Run the following command to train the classification model on one GPU:
```bash
cd src
sh train_class.sh
```

Run the following command to train the detection model on one GPU:
```bash
sh train_detect.sh
```
If you have multiple GPUs, you can change the parameters in .sh file to train multiple models parallelly.

## Inference
Run the following command to inference one test scene:
```bash
cd inference
sh run_inference.sh ../../data/raw_data/test 0157baf3866b2cf9v ../../data/raw_data/predict.csv
```
The shell script takes 3 positional arguments 
The three positional arguments are:
1. `--image_folder` which is the path to the data directory.
2. `--scene_ids` which is the list of the xView3 scene identifier for which you wish to run inference.
3. `--output` which is the path to the prediction output filename in CSV format.

