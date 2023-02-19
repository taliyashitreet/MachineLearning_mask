# MachineLearning_mask
# overview
### DataSet
The dataset is from 'kaggle': https://www.kaggle.com/datasets/shiekhburhan/face-mask-dataset </br>
14,535 images in total. </br>
</br>
**classes:**
    1. incorrect mask (5000 images)
        a."Mask On Chin": These are the images in which masks are put on a chin only.
        the mouth and the nose of a person are visible
        b."Mask On Chin Mouth": In this, the mask is covering the chin and the mouth area.
        The nose of a person is not covered.
        </br>
    2. with mask (4789 images)
        a."Simple With Mask":
        It consists of data samples of face masks without any texture, logos, etc.
        b."Complex With Mask":
        It includes the images of the sophisticated face masks with textures, logos, or designs printed on them.
        </br>
    3. without mask (4746 images)
        a."Simple Without Mask": These are images without any occlusion.
        b."Complex Without Mask":
        It consists of faces with occlusion, such as beard, hair, and hands covering the face.

### Goal
classifying pictures of people are they wearing a mask, and if so - are they wearing it properly?</br>
The classification will be in 3 classes. Each class has 2 types of images, simple and complex.</br>
This is for the data to be more reliable and describe a real situation, so that we can classify correctly not only in training the model but also in the world.
We emphasize that the classification is for 3 classes only, and the division of each class into 2 is to describe a more correct situation, but in terms of the classification,
**the goal is to classify whether a person wore a mask properly / wore a mask poorly / did not wear a mask at all.**

### Our Questions
    1. How to turn the image into a vector that machine learning algorithms can use?
    2. How can we get the best vectors - lose as little information as possible?
    3. Will deep learning give better and faster performance?

### Algorithms
    1. image --> vector:
        a. PCA
        b. Encoder
    2. machine learning:
        a. Logistic Regression
        b. Random Forest
        c. KNN
        d. SVM
        e. Decision Tree
    3. deep learning
        a. CNN - convolution, max-pooling and activation layers.

### Work Process
#### (and difficulties we encountered along the way...)
1. Load Data: we have big data - 14,500 images, each 1024x1024x3.
so, loading it as CSV or something similar was out of the question.
the solution was *ImageDataGenerator*. this is *keras* library for deep learning, which is used to preprocess image data.
The *flow_from_directory* method is used to generate batches of image data from a directory containing subdirectories of images, with each subdirectory representing a class label.


2. Split data to train & test: The data arrived undivided for training and testing,
and since we were working with *ImageDataGenerator* we had to split it already before loading into two separate folders.


3. convert images into vectors, was a complicated mission.
How can we give up most of our features, without losing the important information to classify properly?</br>
we started with *Principal component analysis (PCA)*.
PCA is a popular technique for analyzing large datasets containing a high number of dimensions/features per observation,
increasing the interpretability of data while preserving the maximum amount of information.
At first the model was not able to train at all - even on the train the results were below 50%. After many changes,
we managed to get the model to train and even achieve good results on the test - but unfortunately,
the model reached overfitting! We were unable to prevent this, so we left the model as it is, and moved on to other approaches.
The second approach we tried was *encoder* - encode each image into a vector that most accurately describes it.
This algorithm uses deep learning methods - take the last layer of the deep learning model, and this is our vectors.


4. machine learning algorithms on the vectors:
    * As explained above, the machine learning methods on the *PCA* vectors did not give good results.
    * in the *encoded* vectors, the results was good, although some models got overfitting to.</br>

        * Logistic Regression - Train: 0.93, Test: 0.94 </br>
        * Random Forest - Train: 1.0, Test: 0.94</br>
        * KNN - Train: 0.95, Test: 0.94</br>
        * SVM - Train: 0.94, Test: 0.94</br>
        * Decision Tree - Train: 1.0, Test: 0.91</br>
    --> Random Forest & Decision Tree has overfitting, In our opinion, because they are very strong models by themselves,
   and their combination together with the encoder created a complex model.
   Despite this, the test still got good results, so we left the models as they are.


5. deep learning -CNN.
In this model architecture, we have used 3 Convolutional layers followed by 3 MaxPooling layers,
a flatten layer to convert the 3D tensor into a 1D tensor, a Dropout layer to avoid overfitting,
and finally two Dense layers to classify the images into 3 classes.
the model gave great results too, no overfitting at all!</br>
Train: 0.93, Test: 0.93</br>
# Conclusion
Working with image data required us to open our thinking beyond the limits of machine learning and the algorithms we learned in the course.
The biggest difficulty was working with the large images so that it would run in a reasonable amount of time and still achieve good results. We did a lot of research, and found the approaches we used above.
Our conclusions are that deep learning not only improves the model here, but in certain cases it is almost impossible without it. Except for PCA which does not use deep learning (and did not give good results), all the other methods were based on deep learning, even if we added machine learning to them.

Despite this, it is precisely the machine learning methods that have achieved the highest results! Hence a combination of the two is excellent.

Beyond that, many times we ended up overfitting, despite the many attempts to avoid it. We wondered if it would be better to try to lower the accuracy percentage of the model on the train, even though it led to a decrease in the accuracy percentage of the test? We decided that it was better to have as high an accuracy as possible on the test, so we left it that way.

In addition, beyond the accuracy test, we also tested the recall and precision, to get a broader perspective on the results.

We learned a lot from the project, starting with preprocessing on images, and ending with deep learning and
machine learning methods.
