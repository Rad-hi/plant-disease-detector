# Plant disease detection model creation workshop

## Description

This folder aims at showcasing how one would go about creating a plant disease detection system. Specifically, it's made to guide *you* towards creating your own machine learning model that does the classification of the plant leafs, and how you would start to embed it into an application.

## Train the machine learning model, and obtain the dataset

For the dataset, model creation, and training, this tutorial have been followed: [https://www.kaggle.com/code/atharvaingle/plant-disease-classification-resnet-99-2](https://www.kaggle.com/code/atharvaingle/plant-disease-classification-resnet-99-2).

## How to use

Under `src/` you can find three files:
- `resnet9.py`: this defines the ResNet9 model architecture. As long as you're working with the supplied model, you don't have to touch the content of this file
- `disease_detector.py`: this file is where you might find yourself adding code if you wanted more functionality. It's a wrapper on the machine learning model, and provides a simple interface to deal with it.
- `main.py`: this is an example script on how to use the `DiseaseDetector` class defined in `disease_detector.py`

