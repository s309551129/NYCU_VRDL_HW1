# NYCU_VRDL_HW1
This repository is about HW1 in Visual Recognition using Deep Learning class, NYCU. The main target is to recognize different species of birds.
# Environment
```
python==3.8.5
pytorch==1.9.1
```
# Prepare Dataset
You can download the dataset from codalab in-class competition:
```
https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07#participate-get_starting_kit
```
After downloading, the data directory is structured as:
```
${data}
  +- split_images
     +- training_images
     |  +- 0012.jpg
     |  +- 0013.jpg
     |  +- ...
     +- validation_images
     |  +- 0003.jpg
     |  +- 0008.jpg
     |  +- ...
     +- training_labels.txt
     +- validation_labels.txt
  +- testing_images
  |  +- 0001.png
  |  +- 0002.png
  |  +- ...
  +- training_images
  |  +- 0003.png
  |  +- 0008.png
  |  +- ...
  +- classes.txt
  +- testing_img_order.txt
  +- training_labels.txt
```
```split_images``` folder contain the divided of orginal training images for validation. I already give the .txt files to guide that each training images should belong to which folder.  
# Training
To train the model, run this command:
```
python3 train.py --net model_name
```
You can choose inception_v3, resnet50, resnet101, resnet152 as the backbone model.
# Evaluation
If you want to evaluate a single model, run this command:
```
python3 single_model_eval.py --net model_name
```
If you want to evaluate ensembled model, run this command:
```
python3 ensemble_eval.py
```
After runing the command, you can get answer.txt file. <br> <br>
Please notice that if you want to run ```ensemble_eval.py```, you need to put all models weight into  ```./weight``` folder.
# Pre-trained Models
You can download pretrained weight here:
```
https://drive.google.com/drive/u/0/folders/114oWKAgwHlZyvrjPEpQqW5m0nNkRwU_0
```
After downing all pretrained weight, please put them into ```./weight``` folder.
# Results
Single model (Resnet152): 0.8058 <br>
Ensemble model: 0.8193
# Reference
WS-DAN
```
https://github.com/GuYuc/WS-DAN.PyTorch
```
CAL
```
https://github.com/raoyongming/cal
```
