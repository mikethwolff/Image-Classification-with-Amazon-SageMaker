# Image Classification with AWS SageMaker

Task: Train a pretrained model that can perform image classification by using the AWS Sagemaker profiling, debugger, hyperparameter tuning and other ML engineering practices.

The primary goal of this project is to demonstrate the setup of an ML infrastructure that can facilitate the training of accurate models. It's not just about achieving high accuracy but about understanding and implementing the end-to-end process that makes ML development sustainable, scalable, and efficient. We will use a pretrained image classification model to classify the dog breed, given an image of the dog. This will be done by loading the pretrained model then using transfer learning we adapt the model to the dog breed classification dataset.

## Project Set Up and Installation
Enter AWS  and open SageMaker Studio. Download the starter files. Download/Make the dataset available.

## Dataset

### Access
You can access the data in several ways. You can download the data from "https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip", unzip it on your local machine and upload the data directly to your S3 buckets.
Alternatively you can upload the zip file into your JupyterLab file explorer and then use code in your Jupyter notebook (e.g. train_and_deploy.ipynb) to extract the data and sync the data to your S3 buckets.

### Overview
The provided dataset is a dog breed classification dataset. It contains images of 133 dog breeds divided into training, testing and validation datasets.

## Hyperparameter Tuning
Hyperparameter tuning in Amazon SageMaker is used to automatically optimize the hyperparameters of a machine learning model to achieve the best possible performance on a specific objective metric, such as accuracy, F1 score, or loss. Hyperparameters are configuration settings that control the training process, such as learning rate, batch size, and the number of layers in a neural network. The learning rate in our case ranged beteen 0.001 and 0.1, the batch size values are 32, 64, and 128. We were aiming to maximise accuracy.

<p align="center">
  <img src="./screenshots/screenshot_project_4.JPG" />
</p>
The image above shows the completed hyperparameter jobs. The image above shows the completed training jobs of one hyperparameter job.
<p align="center">
  <img src="./screenshots/screenshot_project_5.JPG" />
</p>

<p align="center">
  <img src="./screenshots/screenshot_project_8.JPG" />
</p>

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
For this experiment whe chose the Resnet-50 (residual neural network). This is a variation of ResNet architecture with 50 deep layers that has been trained on at least one million images from the ImageNet database. The 50-layer ResNet uses a bottleneck design for the building block. A bottleneck residual block uses 1×1 convolutions, known as a “bottleneck”, which reduces the number of parameters and matrix multiplications. This enables much faster training of each layer.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

