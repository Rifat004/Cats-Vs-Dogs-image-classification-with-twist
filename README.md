# Classic Dogs vs. Cats problem in Deep Learning with a twist:

This is a fun deep learning exercise devised by [Abhishek Thakur](https://www.linkedin.com/posts/abhi1thakur_deeplearning-datascience-machinelearning-activity-6797082719605010432-Ne3H) . The problem statements are given below:

1. 🔵 Gather all cat images from here: https://www.kaggle.com/c/dogs-vs-cats
2. 🔵 Gather all dog images from here: https://www.kaggle.com/c/dog-breed-identification
3. 🔵 Divide by dog breeds into two sets: training and test. Note breeds in the test set cannot appear in the training set.
4. 🔵 Use training dog breeds with cat images to train a model
5. 🔵 Evaluate this model on holdout dog breeds
6. 🔵 Is your model able to detect all dogs in your dogs test set?
7. 🔵 What is the accuracy? How many dogs are not classified as dogs?
8. 🔵 Note: you can't use a pre-trained model. Because they already know everything. (I expanded this point and tried three models, one with imagenet weight, another without any pretrained weights, and last one custom model)

## Workflow

**1. Data Preparation:**

- I have directly downloaded the mentioned datasets from kaggle on google colab and stored them in my google drive.

- After extracting the data in specific folder, I took all the cats' data from dogs vs cats dataset (I took from training portion only, as ground truths are only available for those) into 'Cats' folder. From that folder I took 8995 cat images for training("Train/Cat") and 3505 for testing("Test/Cat").

- From "dog-breed-identification" data I took 105 breeds' data out of 120 breeds' data available in the train folder of the original dataset and put them in "Train/Dog" folder. I put the rest 15 breeds' data in "Test/Dog" folder.

- I have created some extra folders for convenience. After completing task, i remove unnecessary folder and made the folder structure like the following.

- So, According to the specification, folder structure:

- Dogs vs Cats

  - Train
  - Test

- The `Train` directory contains two subdirectories:

  - `Cat` with 8995 cat images for training.
  - `Dog` with 8995 dog images for training.

- The `Test` directory contains two subdirectories:
  - `Cat` with 3505 cat images for testing.
  - `Dog` with 1227 dog images for testing.

[Note: Later, I have also renamed the cat images as "cats0,cat01, etc." and dog images as "dog0,dog01,etc. Then stored the file in cloud bucket as zip format. I did not update this part in DataPreparation notebook.]

**2. Connection with google drive and cloud storage and Data download:** I have ensured connection of my colab notebook with cloud bucket and google drive, so that i can download data using gsutil (gsutil is a Python application that lets one access Cloud Storage from the command line). I stored my data in my drive also.

**3. Data Augmentation and splitting into train, validation, and test:**: The downloaded dataset is preprocessed and augmented using the TensorFlow's ImageDataGenerator class. Data augmentation techniques are applied to create variations of the original images, which helps in enhancing the model's generalization capabilities. I set the validation split size 0.15. In the end of this process, I found,

- (Train) 15292 images belonging to 2 classes.
- (Validation) 2698 images belonging to 2 classes.
- (Test) 4732 images belonging to 2 classes.

I used class_mode='sparse'. If we used binary, then the model would check for cat only in this case. In other words, Confusion matrix would end of showing only one column value and 2nd column as 0 ( e.g. 1st row [x 0] , 2nd row [y 0])

**4. Model Architecture:** The CNN model architecture is defined using the Keras API. I have used three different base model architectures.

1. InceptionResNetV2, without pretrained weight
2. EfficientNetV2B2 , with imagenet weight
3. Custom model

Each model is trained in separate notebook. When defining base model, I set include_top=False which means the final fully connected layer at the top of the network, responsible for classification, is not included. As I set sparse mode, in output dense layer, i set 2 units (for two classes) and softmax activation. In the case of InceptionResNetV2 base model, I added some custom layers on top of the model. Global Average Pooling is used to reduce the spatial dimensions of the feature maps while retaining important features. It computes the mean value of each feature map, resulting in a fixed-size feature vector regardless of the input image size. I added dense layer to add non-linearity to the network, helping it learn complex patterns in the data.Batch normalization helps stabilize and accelerate the training process by normalizing the inputs of each layer to have zero mean and unit variance. It also reduces internal covariate shifts and enabling the use of higher learning rates. Dropout is used to prevent overfitting by randomly setting a fraction of input units to zero during each training update.

**5. Creating Callbacks :** Callbacks are utilized during training to monitor the training process and take actions at specific points. Two custom callbacks are used:

- SaveToBucketCallback: This callback saves the model with updated weights and optimizer state to a Google Cloud Storage bucket at specific epoch intervals. This ensures that the model's progress is saved in case the runtime is disconnected during training.

- SaveToDriveCallback: This callback saves the model with updated weights and optimizer state to Google Drive at specific epoch intervals. It provides an additional backup to prevent data loss.

**6. Model compile and fit:** The model is compiled with an optimizer, loss function, and evaluation metrics. For this task, the model is compiled with the Adam optimizer and binary cross-entropy loss. The model is trained using the training dataset while monitoring its performance on the validation dataset. Training proceeds for a specified number of epochs.

**7. Model Evaluation:** I have kept track of training and validation loss in entire training process. After training I have selected the model with best validation Accuracy. Then I have applied the model on the test data.

**8. Result:**

| Model                                           | Accuracy |
| ----------------------------------------------- | -------- |
| InceptionResNetV2 (without pre-trained weights) | 84%      |
| EfficientNetV2B2 (with imagenet weights)        | 80%      |
| Custom Model                                    | 65%      |

## Conclusion:

This project demonstrates the entire workflow of building a deep learning model without pre-trained weights, with imagenet weights, and using custom model for classifying cats and dogs. The use of callbacks and optimization state saving ensures that the model's progress is preserved, allowing for the resumption of training in case of interruptions. The achieved accuracy on the testing dataset reflects the model's capability to generalize and classify images accurately. I have compared accuracy of three models. Among the models I have used, InceptionResNetV2 achieved highest accuracy.

Note: I faced issues when I used model such as MobileNet, it caused overfitting, and val accuracy stuck at 0.5 ven after 10 epochs. I tried many trials and errors including changing model complexity, learning rate, batch_size, but could not solve it. Then I switched model to InceptionResNetV2 (uses more parameters), then the problem did not appear. 
