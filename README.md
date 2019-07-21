# Traffic-signs-classification

# Description
This projects aims to train a classifier that classifies traffic signs.

# Architecture
I've used Convolutional neural networks as my learning algorithm to the dataset.
</br > The architecture is abuout two convolutional layers, the first is about 20 3x3 filters, and the secod is 40 3x3 filters, after each convolutional layer there is max pooling layer of size 2x2, and eventually two fully connected layers of size 200 and 43

# Dataset
The dataset used was german traffic signs, it contains of 26640 different labeled images, 25477 training image, and 1162 testing image, for 43 different classes, the dataset was downloaded from this link: 
</br > http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset


# Training the model
You can simply train the model from the command line using the following command
</br >
```
python3 Training_classifier.py 
```
There is options you may want to set for training your model

Number of epochs to train, the default = 20
```
--epochs = <number>
```

Learning rate, default=0.03
```
--lr = <number>
```

Batch size, default = 64
```
--batch_size = <number>
```

The name of the model you want to save, default = new_model.pt
```
--save = <string of name>
```

If you want to train a pretrained model
```
--model = <string of name>
```
Example
```
python3 Training_classifier.py --epochs=20 --lr=0.03 --batch_size --model="model_sample.pt" --save = "new_model.pt"
```

# Testing the classifier
You can test your model on the dataset from the command line using the following commands:
```
python3 Testing_classifier.p --model = <your model>
```
The only option here is to specify the name of your model file as a string and it is required.

# Classifying
You can use your model to classify different images from the command line using the following commands:
```
python3 classify.py --image = <image file name as string> --model = <name of the model as string>
```
You are required to specify the name of the model you want to use
```
---model = "name of the model"
```

And the image you want to classify
```
--image = "name of the image"
```
