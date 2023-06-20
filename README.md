# Breech Face Multilabel Classifier

## Description 
This repository contains code to train/evalute ResNet50 multilabel classifier model 
to output the probabilities of different labels for shell BF.

## Docker
```shell
docker build --no-cache . -t detect:1.0.0 
```

To work properly script needs model (`--classifier_path` arguments) and configuration file (`--config` file).


#### Docker usage 
```shell
docker run --rm -v ${PWD}/sample_images:/app/data detect:1.0.0 \
--parallel_path saved_models/parallel_classifier.pt \
--dragmark_path saved_models/dragmark_classifier.pt \
--img_path data/2338/averaged.jpg \
-cc data/2338/shell_center.dat \
--out_path result.json \
-c configs/detect_config.yaml 
```


## Inference
To run script without docker use this command
```sh
python detect.py -c configs/detect_config.yaml \
--parallel_path saved_models/parallel_classifier.pt \
--dragmark_path saved_models/dragmark_classifier.pt \
-ip sample_images/2338/averaged.jpg \ 
-cc sample_images/2338/shell_center.dat \ 
-o result.json
```
Also you may consider installing CPU version of pytorch
```shell
python -m pip install torch==1.10.0+cpu torchvision==0.11.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

Makes a prediction for image and returns predicted rotation angle in degrees 

| Argument| Type | Required | Description | 
| -------------------------- | ---  | --- | -------- |
| --config | str | True | Path to detect config <br>`Default value: configs/detect_config.yaml` | 
| --classifier_path | str | True | Path to classifier model<br>`Default value: saved_models/multilabel_classifier.pt` |  
| --img_path | str | True | Path to input image <br>`Default value: sample_images/2338/averaged.jpg`| 
| --center_coordinates | str | True | Path to file with center coordinates <br>`Default value: sample_images/2338/shell_center.dat`|
| --out_path | str | False | Path to .json file to save predicted probabilities <br>`Default value: None`|

## Dataset

### Raw Data 
The original data consists of bullet shell images all assumed to be aligned horizontal. 
<br>
<p align="center">
  <img height=300 src="_readme_imgs/sample.png"/>
</p>

### Data Folder Structure
```
data/      
    |- data_v1/    # Different dataset versions
    |- data_v2/    
          :
    |- data_v7/    
          |- train.csv
          |- test.csv    
          |- tear_drop/  
                 |- bag1/                    
                      |- no_120_map_py_B1_S01_0.25_0.025.jpg    
                      |- no_120_map_py_B1_S02_0.25_0.025.jpg    
                            :
                      |- no_120_map_py_B1_S40_0.25_0.025.jpg                     
                 |- bag2/
                      :
                 |- bag60/    
```

Here's `train.csv`/`test.csv` example file: 

 x | y | path | is_dragmark | is_parallel |
 --- | --- | --- | --- | --- |
 1800 | 1500 | data/data_v7/tear_drop/bag1/no_120_map_py_B1_S01_0.25_0.025.jpg | True | False |

where
- `x` - vertical coordinate from top left corner in pixels
- `y` - horizontal coordinate from top left corner in pixels
- `path` - path to image 
- `is_dragmark` - whether shell has dragmark (True/False)
- `is_parallel` - whether shell has parallel lines (True/False)

### Data Preparation and Augmentation
There is limited training data and likely that the model could overfit to the marks outside of the breechface. 
To prevent this the mid part of the shells will be randomly cropped out and extracted in a circle. 
Then to generate data on rotation angles, the cropped image is randomly rotated.

#### Grayscaled averaged .DNG

As input for our model we use grayscaled images that were made by averaging 16 .dng. 
This was made to get rid of shadows and colors. The aim is to concentrate on shape and edges only.

<p align="center">
  <img height=100 src="_readme_imgs/averaged_dng.png"/>
</p>

#### Hue and Contrast augmentation

Hue augmentation randomly alters the color channels of an input image, 
causing a model to consider alternative color schemes for objects and scenes 
in input images. This technique is useful to ensure a model is not memorizing 
a given object or scene's colors. While output image colors can appear odd, 
even bizarre, to human interpretation, hue augmentation helps a model consider 
the edges and shape of objects rather than only the colors.

<p align="center">
  <img height=100 src="_readme_imgs/aug_hue.png"/>
</p>


#### Coarse dropout

Coarse Dropout is technique to prevent overfitting and encourage generalization. 
It randomly removes rectangles from training images. 
By removing portions of the images, we challenge our models to pay attention 
to the entire image because it never knows what part of the image will be present. 
(This is similar and different to dropout layer within a CNN).

<p align="center">
  <img height=100 src="_readme_imgs/aug_coarse.png"/>
</p>

#### Shifting by x/y axis

As currently we crop BF using center coordinates provided by Sylvain’s 
algorithm which isn’t ideal and has it’s own variance (by Sylvain’s words 
deviation from real center is about 20 pixels). So to increase dataset and 
compensate this deviation we shift our BF by +/-50 pixels by x and y axis.

<p align="center">
  <img height=100 src="_readme_imgs/aug_shifting.png"/>
</p>

## Model Overview
A convolutional neural network called the ResNet50 will be trained to predict the probability of different labels of the breech face.
<p align="center">
  <img src="_readme_imgs/resnet50.png"/>
</p>

- During training images will be passed through the input layer and the model will ouput a single number (predicted probability) for each label.
- To train the model, the output is compared to the actual label of the image and based on the comparison the model weights are adjusted in the direction that will minimize the difference between the output and the label. 

Currently this pipeline works for different BF shapes (tear drop, rectangular and circular), 
but for every shape you have to specify appropriate models and configs:


shape | classifier path | config path |
--- | --- | --- | 
tear drop | - | - | 
rectangular | - | - |
circular | saved_models/multilabel_classifier.pt | configs/detect_config.yaml |

