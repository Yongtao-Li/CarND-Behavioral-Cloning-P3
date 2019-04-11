# **Behavioral Cloning** 

## Yongtao Li

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/simulator.png "simulator"
[image2]: ./examples/iteration1.png "iteration1"
[image3]: ./examples/iteration2.png "iteration2"
[image4]: ./examples/iteration3.png "iteration3"
[image5]: ./examples/model_architecture.png "model architecture"
[image6]: ./examples/left_2016_12_01_13_30_48_287.jpg "left camera"
[image7]: ./examples/center_2016_12_01_13_30_48_287.jpg "center camera"
[image8]: ./examples/right_2016_12_01_13_30_48_287.jpg "right camera"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py: containing the python script to create and train the model
* drive.py: driving the car in autonomous mode 
* model_iteration03.h5: containing the final trained convolution neural network 
* run3.mp4: for recording a successful autonomous driving around 1st track
* writeup_YL.md: for documenting and summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my final model, the car can be driven autonomously around the 1st track in the simulator. Here are the steps about how to do it.

* download the simulator from [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip) for Windows

* unzip the file and run the executable "beta_simulator.exe"

* start my final model that would predict steering command by running the following code
```python
python drive.py model_iteration03.h5
```

* wait until the model started and click on the "AUTONOMOUS MODE" in the following simulator main menu 

![alt text][image1]

#### 3. Submission code is usable and readable

The model.py file contains the code for 

* read the data log file
* read all available images for features and steering command for lables
* set up the Nvidia neural network model
* train and validate the model
* visualize training and validation loss
* save the trained model

The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. I didn't use Python generator, since I used the AWS Deep Learning Instance which has no problem running all available images in the memory.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My final model consists of a convolution neural network which is the same as the [Nvidia CNN](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). It has

* normalization layer (model.py, line 63)
* image cropping layer (model.py, line 65)
* 3 5x5 convolution layer with RELU as activation and depths between 24-48 (model.py lines 67-71)
* 2 3x3 convolution layer with RELU as activation and a depth of 64 (model.py lines 73-75)

Therefore the final model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The train/validate splits has been used with 80% of data for training and 20% for validation. The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 89). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

```python
history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 7, verbose = 1)
```

I didn't use dropout layers since the validation loss wasn't too far away from the training loss and the car could stay on the track pretty well during the whole loop.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 87).

```python
model.compile(loss = 'mse', optimizer = 'adam')
```

#### 4. Appropriate training data

I used entirely the training data from the project resource class and it could be downloaded [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). Then I uploaded the zip file into my AWS DMI and unzipped all the files there for model training.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My design approach could break down to the following steps:

##### 1. work through the work flow and make sure I have all the environments setup correctly

For the first iteration, I chose a very simple regression model and trained it with only images from center camera. As you could see from the following snapshot, it only takes a few seconds to finish training and the losses are very high.  

![alt text][image2]

I still went ahead and run this model in autonomous mode. The car started wabbling at the beginning and couldn't proceed too far away as you could see in the following video. However I was able to make sure I have all the environments setup correctly and I'm ready to improve the model.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=4Cxn_jaKNMs
" target="_blank"><img src="http://img.youtube.com/vi/4Cxn_jaKNMs/0.jpg" 
alt="iteration1" width="240" height="180" border="10" /></a>

##### 2. choose a powerful nerual network model that could significantly improve the test

I chose the [Nvidia CNN](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) because it's designed just for mapping images to steering command. Since I have a good confidence from step 1 in overall work flow, I just need to update the model building part (model.py line 61-85) and keep everything else the same. It doesn't take too much time to train and it shows very good training and validation accuracy in the following snapshot.

![alt text][image3]

I was excited to try this model in the autonomous model. It did pretty well in the first left turn and went over the bridge just fine. It also managed the left turn after the bridge but seems doesn't know how to turn right on the right turn there. So I thought it's probably the time to add more images from left and right cameras. 

<a href="http://www.youtube.com/watch?feature=player_embedded&v=YEg84pRXRRY
" target="_blank"><img src="http://img.youtube.com/vi/YEg84pRXRRY/0.jpg" 
alt="iteration2" width="240" height="180" border="10" /></a>

##### 3. add more images from left and right cameras to train the model

To add more images, I only need to update the code where reading in images and keep all the model the same. Here I used a correction factor 0.1 for left and right camera images. Since it's using more images for training, the model took longer time to train, but the accuracy of the model is still very high.

![alt text][image4]

It's amazing to see that this model just running really smoothly through the whole loop! From the first person view, I could tell it's driving pretty well and staying in the center of the road all the time.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=3xIeSxU2hY0
" target="_blank"><img src="http://img.youtube.com/vi/3xIeSxU2hY0/0.jpg" 
alt="iteration3" width="240" height="180" border="10" /></a>

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 61-85) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:--------------------- |:--------------------------------------------- | 
| Input         		| 160x320x3 RGB image   		    			| 
| Lambda                | normalized to 0-1, output 160x320x3           |
| Cropping              | top 70 bottom 25, outputs 65x320x3           	|
| Convolution 5x5x24    | outputs 31x158x24	     					    |
| Convolution 5x5x36    | outputs 14x77x36                          	|
| Convolution 5x5x48	| outputs 5x37x48        						|
| Convolution 3x3x64	| outputs 3x35x64               				|
| Convolution 3x3x64    | outputs 1x33x64                               |
| Flatten    			| outputs 2112          						|
| Dense                 | outputs 100           						|
| Dense                 | outputs 50            						|
| Dense                 | outputs 10            						|
| Dense                 | outputs 1              						|

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image5]

#### 3. Creation of the Training Set & Training Process

As I mentioned earlier, I only used the training data from project resource, as you could see the examples below. For the final model, I used all the images from center, left and right cameras. I could also flip images to get more data for training, but didn't end up doing so because the test already shows great behavior.

| left camera      		|     center camera	    | right camera   		| 
|:---------------------:|:---------------------:|:---------------------:| 
|![alt text][image6]    |![alt text][image7]    |![alt text][image8]    |

I finally randomly shuffled the data set and put 20% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. It seems that the validation loss isn't too far away from training loss so that I don't think the finall model is over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary. From previouls loss plot, 7 epochs are probably more than enough for model training, since the validation losses are not decreasing any more. 

### Is the car able to navigate correctly on test data?

Yes! The car is able to navigate correctly on the first track in the simulator autonomously.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=3xIeSxU2hY0
" target="_blank"><img src="http://img.youtube.com/vi/3xIeSxU2hY0/0.jpg" 
alt="iteration3" width="240" height="180" border="10" /></a>
