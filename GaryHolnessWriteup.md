# **Traffic Sign Recognition** 

## Writeup

### Gary Holness

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
### Data Set Summary & Exploration

The German road sign data set represents 32 x 32 pixel color images of 43 different categories of road sign.

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Using numpy code, I measured the data set, deciding upon the following sizes for the training/test/validation sets.

* The size of training set is 34977 examples
* The size of the validation set is 4410 examples
* The size of test set is 12630 examples
* The shape of a traffic sign image is 32 x 32 x 3 (i.e. 32 x 32 pixel RGB color image)
* The number of unique classes/labels in the data set is 43 class labels.

#### 2. Include an exploratory visualization of the dataset.

An exploratory visualization of the data set is included.  This consists of histograms depicting
the prior distribution over class labels for the training set, test set, and validation set.
An objservation is that the three prior distributions are similar in shape with the training
and test set being most similar.  While the validation set comes close to the training/testing
set, it is a little differently shaped.  This is most likely due to the relatively small number
of exemplars (4410 instances) in the validation set in contrast to the training set (34799 instances)
and the testing set (12630 instances).  As the number of instances is larger, it more easlily represents
the distribution.  With a smaller number of examples, such as with the validation set, one doesn't
have as true a representation of the distribution.  After all we have 43 classes and need a sufficiently
large number of samples (examples) to get a good representation of the distribution over 43 class labels.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I began by converting the images to grayscale.  The data-sets (training, testing, validation) are thought of as
a stack of images where each slice within the stack is a single 32 x 32 x 3 image.  I weight the RGB color channels
equally by multiplying by the vector [0.33, 0.33, 0.33]. This takes each 32 x 32 x 32 and applies the multiplication
to each pixel (32 x 32) resulting in a 32 x 32 grayscale image.  This is done for the training, testing, and validation
sets.   

I followed this by normalizing the grayscale pixel intensities.  This was done by subtracting 128 from each pixel value
and dividing by 128 essentialy shifting the mean to 0 and the deviation to the range [-1,1] on the Real Number line.
I randomly select an image, display it in color and display it's grayscale normalized representation.  Conversion to
grayscale was done to simplify the representation and normalization was done to clean up the data.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I implemented two version of the architecture, LeNet and LeNet1.  I ended up using LeNet1 because of better performance.
The diference between the two is that LeNet1 implements the fully connected layer explicitly where LeNet uses the TensorFlow
API to create it.  By creating it explicitly, I had control over the weight initialization, that is to use truncated_normal.

The second major implementation stems from experiments training the classifier.  I found that the validation accuracy
increased then decreased over training epocs.  This meant the network was overfitting.  To address this, I included
regularization in the form of dropout at each layer.  I found a relatively small bit of regularizaiton was warranted
(as is typical) so the keep probability was set to 90%, meaning only 10% dropout.

LeNet1 description is

Input Layer 32 x 32 x 1  grayscale image

Layer 1:  Convolution input 32 x 32 x 1  output 28 x 28 x 6  
          shape= 5 x 5 x 1 x 6    stride = 1 x 1 x 1 x 1  padding=VALID
          activation = relu
          dropout keep prob= 90%

          Max Pooling input = 28 x 28 x 6  output= 12 x 14 x 6
          size= 1 x 2 x 2 x 1  stride = 1 x 2 x 2 x 1  padding= VALID

Layer 2:  Convolution input 12 x 14 x 6  output 10 x 10 x 6
          shape = 5 x 5 x 6 x 16    stride = 1 x 1 x 1 x 1  padding= VALID
          activation= relu
          dropout keep prob = 90%

          Max Pooling input= 10 x 10 x 16  output = 5 x 5 x 16
          
          Flatten input= 5 x 5 x 16  output= 400

Layer 3:  Fully Connected  input = 400  output= 120
          activation = relu
          dropout keep prob = 90%


Layer 4:  Fully Connected input = 120   output = 84
          activation= relu
          dropout keep prob= 90%
         

Layer 5:  Fully Connected input = 84  output=43
          output logits 43 of them 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I minimized the mean cross entropy between the one-hot encoded ground truth class label
and the model's one-hot encoded logits through softmax.  I used a batch size of 200 instances over 20 epocs.
The learning rate was 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

To create the validation set, I split off 20% of the training set in each batch after shuffling the training set.
The randomization is intended to break any structural dependencies.  Using validation set consisting of 20% of
training set strikes a good balance between estimating generalization error while giving training enough examples
(i.e. the 80%) to find a good model.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 67.70%
* test set accuracy of 96.68%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I chose LeNet because it was used for MNIST classification

* What were some problems with the initial architecture?

The architecture overfit.  In addition using truncated_normal weight initialization made it better

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I did the fully connected layers myself so that I could initialize with truncated_normal.
Also I added the dropouts for regularization because experiments showed overfitting.

* Which parameters were tuned? How were they adjusted and why?
I increased the number of epochs and the batch size.  This was done because I felt the model needed
more inputs in order to find better model.  Additionally since validation set was carved out of the
training set, I needed bigger batch size.   The number of epochs was set to as large aso 200 because
i wanted to see how it behaved over a long run.  After tuning the learning rate starting with very 
small value of 0.00001 and icreasing by factor of 5, I settled upon 0.001 because it worked well.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The convolution layer is good with recognizing the edge information contained in the sign images.
The dropout is useful in regularizing the model in effort to mitigate overfitting.

If a well known architecture was chosen:
* What architecture was chosen?

The LeNet Architecture was chosen

* Why did you believe it would be relevant to the traffic sign application?

LenNet is good with image classification problems. Since it worked on MNIST which contains
mostly edge information, it should also work well with road signs.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The validation accuracy serves as an estimate for test accuracy.  As validation accuraccy increases over a sequence of epocs,
this tells you that with more examples, generalization is improving. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I started out by randomly selecting 43 road sign images from the test set. One example
from each of the 43 classes was randomly selected to form what I call
the exemplars. This was not the correct approach.  I am curious though, had I set asside
43 road sign images from the test set, how would it have performed. Perhaps fairly well.

Nonetheless, I went back and found 5 images of German road signs using Google image search.
These images were of varying resolutions, so usinging the Preview application on MacOS,
I changed each image to 32 x 32 pixel color images.  In many cases, this required changing
the original aspect ratio. The effect of this was that the scaled down images were warped
from their original version.  Moreover, The road sign images have varying backgrounds including
fields, roadways, and buildings.  Features associated with varying background also impacted
the content of the web images.   These proved to be very difficult to classify.  The reason
for this stems from the general content of the training set.  The road sign images are, for
the most part, croppped so that the road sign occupies most of the image.  This is certainly
not the case with the road sign images I obtained from the web.  A good additional processing
step would be to segment the foreground (i.e. road signs) from the background, and artificially
rotate and scale the road sign in the foreground so that it appears centered and occupies
most of the 32 x 32 pixel image.  Additionally, while 32 x 32 pixel makes the problem easier
from a network architecture perspective, that is not much information when converted to grayscale.
More information would be provided if full color were employed versus using normalized
grayscale.  In addition having bigger images, say 620 x 480 would perhaps give the network
opportunity to describe features discriminating background from foreground, essentially
picking out the road side features "in context."

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The accuracy on these exemplars was 74.2%
This was worse than the performance on the test set.  The reason for this, I speculate is that 
with only a single example from each of the 43 classes, it is entirely possible that randomization
picked a "difficult" isntance.

The accuracy on the web-obtained roadsign images was an abismal 0%.
The reason for this is due
to the drastic difference between the web versions and the training set versions.  The web
versions were actual road sign images as one would see them in context.  That is there was
alot of background including grass, buildings, etc.   Moreover, relatively few of the image
pixels consisted of road sign whereas for the training set, the most of the image pixels
consisted of road sign.

When predicting the web-obtained road sign images, the predictions were as follows

image 1:   Speed Limit 70 km/hr (label 4)   top 5 predicted labels: [35 34 12 10 29]
image 2:   Speed Limit 100 km/hr (label 7)  top 5 predicted labels: [17 13 26 12 14]
image 3:   Children crossing (label 28)     top 5 predicted labels: [ 0  1 37 29 18]
image 4:   Speed Limit 30 km/hr (label 1)   top 5 predicted labels: [ 2  4  1 37 39]
image 5:   Stop (label 14)                  top 5 predicted labels: [40 12 14 33  2]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

From observation, when I selected images from test set, I noticed for many cases the softmax probabilities that
the "winning" class label has a probability whose exponent was e-01 where the next highest probability had a
far smaller exponent like e-05.  This means the model is very certain about the predicted label because the
2nd, 3rd, etc. choice have softmax probabilities that are multiple orders of magnitude smaller.

Contrast this with the web-obtained roadway images.  The model is less certain.  I observed that the 
softmax probabilities are only a single order of magnitude off from one another.  The top label has
a softmax probability whose exponent is e-01, the 2nd label has softmax probability with exponent e-02,
followed by 3rd with e-03 etc.  In one case the softmax probabilities had exponents
[e-01  e-01 e-02 e-03 e-03].  The fact that the top choice and 2nd choice were very close (same exponent e-01)
means that the model was less certain.  A more certain model would have a far larger probability assigned
to the top label (predicted label).

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Did not do this one.
