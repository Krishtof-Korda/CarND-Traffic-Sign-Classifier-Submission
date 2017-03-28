#**My First Traffic Sign Classifier Using Deep Learning** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Rotated_Example.png "Rotated"
[image2]: ./examples/Grayscale_Example.png "Grayscaling and Normalization"
[image4]: ./internet-traffic-signs/Priority-road.jpg "Traffic Sign 1"
[image5]: ./internet-traffic-signs/Right-turn.jpg "Traffic Sign 2"
[image6]: ./internet-traffic-signs/Road-Work.jpg "Traffic Sign 3"
[image7]: ./internet-traffic-signs/Speed-30.jpg "Traffic Sign 4"
[image8]: ./internet-traffic-signs/Straight-right.jpg "Traffic Sign 5"
[image13]: ./Softmax_Figs/Go%20straight%20or%20right.png "Traffic Sign 1" 
[image9]: ./Softmax_Figs/Priority%20road.png "Traffic Sign 2"
[image11]: ./Softmax_Figs/Road%20work.png "Traffic Sign 3"
[image12]: ./Softmax_Figs/Speed%20limit%20(30km%20per%20h).png "Traffic Sign 4"
[image10]: ./Softmax_Figs/Turn%20right%20ahead.png "Traffic Sign 5"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Krishtof-Korda/CarND-Traffic-Sign-Classifier-Submission/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fifth code cell of the IPython notebook. 

Below is a list of the quantity of each traffic sign in the training data set. I also plotted out the first occurence of each sign in the dataset and titled the plot with its SignName and ClassID from the signnames.csv file. This helped me to know exactly which sign is classified by which ClassID. This was important for when I found my own signs from the internet so that I could create my y_internet labels array for accuracy checking. 

 Count 180  =  Speed limit (20km/h)  for ClassID =  0
 
 Count 1980  =  Speed limit (30km/h)  for ClassID =  1
 
 Count 2010  =  Speed limit (50km/h)  for ClassID =  2
 
 Count 1260  =  Speed limit (60km/h)  for ClassID =  3
 
 Count 1770  =  Speed limit (70km/h)  for ClassID =  4
 
 Count 1650  =  Speed limit (80km/h)  for ClassID =  5
 
 Count 360  =  End of speed limit (80km/h)  for ClassID =  6
 
 Count 1290  =  Speed limit (100km/h)  for ClassID =  7
 
 Count 1260  =  Speed limit (120km/h)  for ClassID =  8
 
 Count 1320  =  No passing  for ClassID =  9
 
 Count 1800  =  No passing for vehicles over 3.5 metric tons  for ClassID =  10
 
 Count 1170  =  Right-of-way at the next intersection  for ClassID =  11
 
 Count 1890  =  Priority road  for ClassID =  12
 
 Count 1920  =  Yield  for ClassID =  13
 
 Count 690  =  Stop  for ClassID =  14
 
 Count 540  =  No vehicles  for ClassID =  15
 
 Count 360  =  Vehicles over 3.5 metric tons prohibited  for ClassID =  16
 
 Count 990  =  No entry  for ClassID =  17
 
 Count 1080  =  General caution  for ClassID =  18
 
 Count 180  =  Dangerous curve to the left  for ClassID =  19
 
 Count 300  =  Dangerous curve to the right  for ClassID =  20
 
 Count 270  =  Double curve  for ClassID =  21
 
 Count 330  =  Bumpy road  for ClassID =  22
 
 Count 450  =  Slippery road  for ClassID =  23
 
 Count 240  =  Road narrows on the right  for ClassID =  24
 
 Count 1350  =  Road work  for ClassID =  25
 
 Count 540  =  Traffic signals  for ClassID =  26
 
 Count 210  =  Pedestrians  for ClassID =  27
 
 Count 480  =  Children crossing  for ClassID =  28
 
 Count 240  =  Bicycles crossing  for ClassID =  29
 
 Count 390  =  Beware of ice/snow  for ClassID =  30
 
 Count 690  =  Wild animals crossing  for ClassID =  31
 
 Count 210  =  End of all speed and passing limits  for ClassID =  32
 
 Count 599  =  Turn right ahead  for ClassID =  33
 
 Count 360  =  Turn left ahead  for ClassID =  34
 
 Count 1080  =  Ahead only  for ClassID =  35
 
 Count 330  =  Go straight or right  for ClassID =  36
 
 Count 180  =  Go straight or left  for ClassID =  37
 
 Count 1860  =  Keep right  for ClassID =  38
 
 Count 270  =  Keep left  for ClassID =  39
 
 Count 300  =  Roundabout mandatory  for ClassID =  40
 
 Count 210  =  End of no passing  for ClassID =  41
 
 Count 210  =  End of no passing by vehicles over 3.5 metric tons  for ClassID =  42
 

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 6th code cell of the IPython notebook.

The first step in pre-processing that I chose was a combination of grayscaling and normalization. The grayscaling is accomplished using the formula gray = (0.299*red + 0.587*green + 0.114*blue). Normalization is accomplished by finding the minimum and maximum values of each image and scaling the gray value in the range -1 to 1, centered at the origin. 

Here is an example of a traffic sign image after and before grayscaling and normalization. Note the colorbar for the grayscaled/normalized image is perfectly centered with limits -1 and 1. 

![alt text][image2]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the second code cell of the IPython notebook.  

To cross validate my model, the data was split into training and validation sets.

Training data: 
Shape of grayed colors =  (34799, 32, 32)
Grayscaled shape =  (34799, 32, 32, 1)
Features have been grayscaled and normalized

Validation data: 
Shape of grayed colors =  (4410, 32, 32)
Grayscaled shape =  (4410, 32, 32, 1)
Features have been grayscaled and normalized

Test data: 
Shape of grayed colors =  (12630, 32, 32)
Grayscaled shape =  (12630, 32, 32, 1)
Features have been grayscaled and normalized

The seventh code cell of the IPython notebook contains the code for augmenting the data set.

After reading the section for standing out, I implemented random 90*k degree rotation on all the images in the training set to improve accuracy of the network. Ultimately, I found that the rotations reduced the overall validation accuracy over the same number of epochs. Because of this I decided to remove it. I believe it didn't help the network because there are no examples that I had seen where the sign was that drastically rotated. Therefore, it was not useful in improving accuracy. Had there been some examples with drastic rotation this could have help to properly classify those.

Below is an example of the rotation augmentation that I decided against:

![alt text][image1]

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 10th cell of the Ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x18 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 6x6x40 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 3x3x40 				|
| Flattened	    | outputs 360     									|
| Fully connected		| 120        									|
| Fully connected		| 90        									|
| Fully connected		| 43        									|
| Softmax with Cross Entropy				| 43        									|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 14th cell of the Ipython notebook. 

To train the model, I used an AdamOptimizer function with 60 epochs or batch size 128. I tried many differnet combinations of epochs and batch sizes and found this to be the best in terms of time to train and accuracy achieved. I used a learning rate of 0.001. Again after trying rates ranging from 0.005 to 0.0005, I found 0.001 to be the best balance of speed and performance. 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 13th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.976 
* test set accuracy of 0.951

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  The first architecture I tried was obviously what we learned in the LeNet lab. That achieved fairly good performance out of  the box at 0.92 validation accuracy. Obviously I wanted to improve upon that and try new things.
  
* What were some problems with the initial architecture?
  This architecture did not implement grayscale or normalization and took a long time to train since the images were ill formed for optimization. It also didn't have any dropout implemented. 
  
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

  The first adjustments were the grayscaling and normalization to speed up convergence of the model. The next improvement restructuring the layers to add two back to back convolutions before pooling. This was my biggest gain in accuracy after normalization. After this I added dropout between each fully connected layer to improve accuracy a little further. I never really saw any over or under-fitting through any of the architectures. My biggest reason for trying different architectures was to improve validiation accuracy. 

* Which parameters were tuned? How were they adjusted and why?
  I played a lot with learning rate and number of epochs. The learning rate seemed to be very hard to dial in perfectly. It seemed to have too much momentum or too little. I was never just right. The epochs was easier to tune, I quickly found that not much learning was happening after 35 epochs. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  I believe convolution is the most important design choice for this application because it adds context to the adjacent pixel in the image. It also helps speed up learning of the model. Max pooling was another choice which consolidated pixels in a certain area taking advantage of the fact that from pixel to pixel there shouldn't be drastic change in the real world. I also found dropout to improve accuracy by about 3%. I attributed that to the model never overweighting a particular activation, thereby being more tolerant of image differences for the same traffic sign. 

If a well known architecture was chosen:
* What architecture was chosen?
  I chose LeNet to start with and then added some convolution layers between pooling.
  
* Why did you believe it would be relevant to the traffic sign application?
  Based on the lessons LeNet seemed to be an obvious choice for image data of the traffic signs. Especially considering the common shapes involved with a traffic sign. 
  
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  The accuracies on the training (0.999) and validation (0.976) are within 2.3% of each other, while the test accuracy (0.951) is within 2.5% of validation. This led me to believe that the model is very good at determining signs its never seen in training.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because of the pole behind the sign creating contrast against the sky which might be interpreted as part of the sign.

The second image has some noise in the background which might make for difficult contrast to the sign.

The third image is taken from a low angle and distorts the perspective which might prove difficult. Also once sized to 32x32 the man shoveling the pile looks like it could be anything from children crossing to pedestrian. 

The fourth image I included the word 'zone' on the sign to see if that would trip it up, but surprisingly it didn't. It is very similar though to any speed limit sign. That was apparent in the models uncertainty on this image in the softmax probabilities.

The fifth image has contrasted background and part of the sign below it. This could have led to confusion for the model but based on the softmaxes it was very certain. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

Here are the results of the prediction:

Predictions =  [12 33 25  1 36]
Actuals = [12 33 25 1 36]

Prediction =  Priority road  
Actual Sign =  Priority road

Prediction =  Turn right ahead  
Actual Sign =  Turn right ahead

Prediction =  Road work  
Actual Sign =  Road work

Prediction =  Speed limit (30km/h)  
Actual Sign =  Speed limit (30km/h)

Prediction =  Go straight or right  
Actual Sign =  Go straight or right


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.1%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for visualizing softmax probabilities on my final model is located in the 21st cell of the Ipython notebook.

For the first image, the model is very sure that this is a 'priority road' (probability of 0.999), and the image does contain a 'priority road' sign. The top five soft max probabilities are shown in the figure:

![alt text][image9] 

For the second image, the model is very sure that this is a 'turn right ahead' (probability of 0.999), and the image does contain a 'turn right ahead' sign. The top five soft max probabilities are shown in the figure:

![alt text][image10] 

For the third image, the model is very sure that this is a 'road work' (probability of 0.999), and the image does contain a 'road work' sign. The top five soft max probabilities are shown in the figure:

![alt text][image11] 

For the fourth image, the model is sure but not very that this is a 'speed limit (30 km/h)' (probability of 0.51), and the image does contain a 'speed limit (30 km/h)' sign. The top five soft max probabilities are shown in the figure:


![alt text][image12] 

For the fifth image, the model is very sure that this is a 'Go straight or right' (probability of 0.999), and the image does contain a 'Go straight or right' sign. The top five soft max probabilities are shown in the figure:

![alt text][image13]

Overall I am very satisfied with the results of my model. I am sure there are many shortcomings that I have not yet explored but for the project at hand I believe it to be a good classifier. I did notice that once I visualized the featuremaps, that I may have 'zoomed' in too much with the layers of convolution. If I did this again, I would probably choose 'same' padding to maintain image size for the first few layers and let the pooling layers handle the downsizing. 

Once again, I applaud the creators of this project. I can see the immense effort that went into everything from the notebook to this template for the report. Thank you for making learning easier and fun!
