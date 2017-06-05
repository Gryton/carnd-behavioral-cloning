#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./writeup_photos/center_2016_12_01_13_31_12_937.jpg "Normal image"
[image3]: ./writeup_photos/center_2017_06_01_20_33_36_197.jpg "Recovery Image"
[image4]: ./writeup_photos/center_2017_06_01_20_33_36_766.jpg "Recovery Image"
[image5]: ./writeup_photos/center_2017_06_01_20_33_37_052.jpg "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I mostly used NVIDIA architecture to create my model. Before pushing data into network I normalize it using Keras lambda layer
then crop to leave only road, and get rid of the rest of environment. I used RELU activations to introduce nonlinearity.

I thought it might be best choice, as network is not complicated and also used for same purposes. First
I thought of using coma.ai architecture, but I found it rather complicated and difficult to implement
directly in Keras, without Keras functions overloading. 


####2. Attempts to reduce overfitting in the model

The only difference that's not mentioned in NVIDIA architecture is adding dropout after first
convolutional layer and first dense layer, to reduce overfitting. I used example test data, which contains
few driving laps, also to reduce overfitting. Each time I've added more data to training set I've added
the same part driven few times with different driving lines. Shuffling data also should reduce overfitting.

####3. Model parameter tuning

The only manual model parameter tuning was done for dropout layer, where I finally ended with 0.5.
I haven't tuned learning rate manually as I used adam optimizer and loss as mean square error.

####4. Appropriate training data

To have a fast start I used example data, as I'd have to spend lot of time to record enough data.
It looked that it's enough data for quite nice autonomous model, especially including left and
right camera, but additional recovery data was essential. Without gathering additional data everything
looked good, until reaching critical section, where car wasn't in the center of track and suddenly getting off.

I knew that I would need data for getting back to the center of track, and to get rid of the car trying to go
rather left then right in whole track, so I've added image flipped left to right, and images from left and right
camera, to which I've added (or subtracted) bias 0.3 for angle. I started with 0.5, but car was too nervous in
when getting a little closer to side of track, so I decreased it to 0.3.

###Model Architecture and Training Strategy

####1. Solution Design Approach

Instead trying to create whole own architecture I rather wanted to test well known architectures,
and find the best one. 
I started with LeNet, and I was positively surprised that it worked really nice.
Car didn't drive like real driver, but I thought that might be good way to go, if other nets will consume
too much computing power for my gear. Next, I wanted to try comma.ai network, but I found it to complicated
to implement it in Keras without overloading Keras features. So next, I tried complete NVIDIA network,
and it seemed to be not an overkill for my gear, and it gave me really well results, car drives really
smooth, till the bridge, where it suddenly crashed after getting of the center line.
So I thought that I have my favourite, and now I just have to make it whole lap.

When I saw much difference between data for training and validation set I thought that I will add dropout layers.
I decided to make it iterational, so I added dropout after first convolution layer and trained the net, then
added dropout after first dense layer, and when I got similar error for training and validation set I thought
that might be ok for getting to next step.

My later approach was to record recovering and normal drive line in every place I would see problems with. So first was the
bridge, but after adding few recovery drives to the center of bridge at entry, recovery when going into edge,
and few lines ending in center of bridge when getting off I finally made it. Next problem was just after the
bridge in sharp corner, that ended in gravel shortcut. I managed to go with car by the gravel shortcut, and
nearly came back to track, but I thought that wasn't what was needed for this project. Adding recovery when getting
close to gravel road helped well, and then I was able to make full lap without touching kerbs.


####2. Final Model Architecture

The final model architecture (model.py lines 57-75) consisted of a NVIDIA network + dropout layers.
I used Lambda and Cropping2D layers to normalize and crop data to show only the road, then I have convolutional
layer with 24 depth and kernel 5 + dropout and RELU activation after it. Then I have 2 convolutional layers with
kernel 5, and depth 36 and 48 respectively. Then I have 2 convolutional layers with kernel 3 and depth 64.
Then I flatten architecture. Next goes connected layers with depth 100, 50, 10, 1, and dropout + RELU activation
after first of them.

####3. Creation of the Training Set & Training Process

Good driving line was used from example data. Here is picture how look normal driving line from example set:

![alt text][image2]

Below are images where is recovery when car goes straight to gravel road, and then is radically turned to go back to the track.

![alt text][image3]
![alt text][image4]
![alt text][image5]


After all, I had 8877 samples, with 3 gathered images per each (center, left and right) and 1 added during
data augmentation (by flipping center image). I'm not sure if flipping really makes huge difference, as track
was driven also backward, so maybe just recording few additional sequences would be enough,
 but I thought that I will stay with flipping, as I had full clean lap then. As data was splitted training:validation
 with coefficient 80:20 it gave me 28 406 samples for training and 7 102 samples for validation. Data was shuffled
 for training and validation set, and then also during training dataset used for it was shuffled.
 
 I tried to manually tune data, and I've added progressive bias in my recovery drives, so I thought that should
 end up with faster car recovery, but it seemed to even lower car ability to start fast turns, so I got back to
 originally gathered data.

I used 3 epochs, as I saw then minor training gain, and also training on such set for 3 epochs lasts over 10 minutes
on my NVIDIA GeForce 940MX, so I didn't want to prolong it, as it'd be really not nice to tune something with
longer time to see results of tuning or addtional data.

After all, I successfully created network that has ability to drive a simulator for a whole clean lap. I tried
also increasing speed of simulated car, but it seems to be a bit to nervous to make it with 30 mph - probably I
would need to optimize a bit of recovery and biases for right and left camera images to be able to make clean
lap with such speed.