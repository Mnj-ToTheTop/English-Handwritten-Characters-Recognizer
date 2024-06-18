# English-Handwritten-Characters-Recognizer

## STEP - 1: Loading the Dataset

First step I took was to download the dataset.

The labels and the images names were stored in a CSV (Comma Separated Value) file. I used the `read_csv()` fuction of the Pandas library to achieve this.

![image](https://github.com/Mnj-ToTheTop/English-Handwritten-Characters-Recognizer/assets/153396359/ba7c5dd8-43c2-4bc5-bc30-f00c4c42d127)


Looking at the data it can be noted that the first column contains the Image path while the 2nd contains the True Label of the image. There are a total 3409 images.

I then created a Python list `labels` and `filenames` to store the image path of each image along with its corresponding label.

## STEP 2: Preprocessing
To make sure all the images are of the same type and that all the images have the proper input size for our model, I used the `imread()` function of OpenCV library to read the image and the `resize()` function of the same to resize the image to the correct size.

Then I converted both the labels list and filenames list to Numpy arrays. I also used the `LabelEncoder` from the scikit-learn library to encode the the labels to integers (as some of the labels are characters)

## STEP 3: Separating the Train set and Test set
To split the dataset into train and test set, I have used the `train_test_split()` of the scikit-learn library. Once the the train set and test set were obtained, the labels were One-Hot-Encoded using the `to_categorical()` function of Keras.

## STEP 4: Model Building
The model contains 3 Convolution Layers and Classification Layer.

### *LAYER - 1*
This layer contains two `Conv2D()` layers one with 64 filters and the other with 32 filters, with the kernel size = 3 (i.e. 3x3). Following this, a Max Poooling layer is in place, which gives the output shape of `32x32x32`.

### *LAYER - 2*
This layer contains two `Conv2D()` layer each with 64 Filters and a kernel size of 3 (i.e. 3x3). This is followed by Max Pooling with a size of 2x2 which results in an output size of `8x8x64`.

### *LAYER - 3*
This layer is identical to Layer - 2 and results in an output size of `8x8x64`.

`NOTE: I have used padding to avoid shrinking of the data.`
`NOTE 2: The activation function used is Rectified Linear (ReLu)`

### CLASSIFICATION LAYERS:
#### *FLATTEN*
This layer is used to flatten the matrix to a vector. On flattening the matrix the total elements we got: `8 * 8 * 64 = 4096`
#### *DROPOUT LAYER - 1*
To reduce overfitting, we are dropping 0.25 of the neurons.
#### *DENSE LAYER - 1*
A fully connected layer of nuerons with output of shape of `512`
#### *DROPOUT LAYER - 2*
Dropping 0.50 of the nuerons
#### *DENSE LAYER - 2*
Since we have 62 classes to classify, we have kept the the number of nuerons to `62`

Dataset Taken from:

@InProceedings {deCampos09,
author = "de Campos, T.~E. and Babu, B.~R. and Varma, M.",
title = "Character recognition in natural images",
booktitle = "Proceedings of the International Conference on Computer
Vision Theory and Applications, Lisbon, Portugal",
month = "February",
year = "2009",

Looking at the data it can be noted that the first column contains the Image path while the 2nd contains the True Label of the image.

I then created a Python list labels and filenames to store the image path of each image along with its corresponding label.

STEP 2: Preprocessing
To make sure all the images are of the same type and that all the images have the proper input size for our CNN, I used the imread() function of OpenCV library to read the image and the resize() function of the same to resize the image to the correct size.

Then I converted both the labels list and filenames list to Numpy arrays. I also used the LabelEncoder from the scikit-learn library to encode the the labels to integers (as some of the labels are characters)

STEP 3: Separating the Train set and Test set
To split the dataset into train and test set, I have used the train_test_split() of the scikit-learn library. Once the the train set and test set were obtained, the labels were One-Hot-Encoded using the to_categorical() function of Keras.

STEP 4: Model Building
The model contains 3 Convolution Layers and Classification Layer.

LAYER - 1
This layer contains two Conv2D() layers one with 64 filters and the other with 32 filters, with the kernel size = 3 (i.e. 3x3). Following this, a Max Poooling layer is in place, which gives the output shape of 32x32x32.

LAYER - 2
This layer contains two Conv2D() layer each with 64 Filters and a kernel size of 3 (i.e. 3x3). This is followed by Max Pooling with a size of 2x2 which results in an output size of 8x8x64.

LAYER - 3
This layer is identical to Layer - 2 and results in an output size of 8x8x64.

`NOTE: I have used padding to avoid shrinking of the data.`
`NOTE 2: The activation function used is ReLu`

## STEP 5: Training and Testing the model
The model was trained using the `model.fit()` function. 30 Epoches were used.
On evaluating the model using the test set, and the function `model.evaluate()` we can see the model has an accuracy of 75.806%

## IMPROVEMENTS THAT CAN BE MADE :
1. A larger data set can be used to train the model more.
2. A larger number of epoch can be used.
3. Changes in the model can also be done to improve the model.

Dataset Taken from:

  @InProceedings {deCampos09, <br />
  author    = "de Campos, T.~E. and Babu, B.~R. and Varma, M.", <br />
  title     = "Character recognition in natural images", <br />
  booktitle = "Proceedings of the International Conference on Computer <br />
  Vision Theory and Applications, Lisbon, Portugal", <br />
  month     = "February", <br />
  year      = "2009", <br />
}

Kaggel Link: https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset
