# Image Classification with AWS SageMaker

Task: Train a pretrained model that can perform image classification by using the AWS Sagemaker profiling, debugger, hyperparameter tuning and other ML engineering practices.

The primary goal of this project is to demonstrate the setup of an ML infrastructure that can facilitate the training of accurate models. It's not just about achieving high accuracy but about understanding and implementing the end-to-end process that makes ML development sustainable, scalable, and efficient. We will use a pretrained image classification model to classify the dog breed, given an image of the dog. This will be done by loading the pretrained model then using transfer learning we adapt the model to the dog breed classification dataset.

## Project Set Up and Installation

- Clone Github repository or download the starter files.
- Enter AWS  and open SageMaker Studio.
- Download and make the dataset available.
- Create notebook instance, load files to JupyterLab, start developing notebook.

<p align="center">
  <img src="./screenshots/screenshot_project_21.JPG" />
</p>

This project contains the following files: 

[**train_and_deploy.ipynb**](./train_and_deploy.ipynb): Jupyter notebook used in JupyterLab to perform the complete workflow from environment setup to model deployment.<br>
[**hpo.py**](./hpo.py): Python script file used for hyperparameter optimization<br>
[**train_model.py**](./train_model.py): Python script used for model training and deployment.<br>
[**inference.py**](./inference.py): Python script prepares the model for inference.<br>

### Project Steps

The following project steps will be performed
- envionment setup,
- data upload and exploration,
- hyperparameter tuning,
- model profiling and debugging,
- model deploying and prediction.

## Dataset

### Access

You can access the data in several ways. You can download the data from "https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip", unzip it onto your local machine and upload the data directly to your S3 buckets.
Alternatively you can upload the zip file into your JupyterLab file explorer and then use code in your Jupyter notebook (e.g. [**train_and_deploy.ipynb**](./train_and_deploy.ipynb) ) to extract the data and sync the data to your S3 buckets.

### Overview

The provided dataset is a dog breed classification dataset. It contains images of 133 dog breeds divided into datasets for training, testing and validation. In total more than 8000 image files.

<p align="center">
  <img src="./screenshots/screenshot_project_18.JPG" /><img src="./screenshots/screenshot_project_19.JPG" />
</p>

## Hyperparameter Tuning

Hyperparameter tuning in Amazon SageMaker is used to automatically optimize the hyperparameters of a machine learning model to achieve the best possible performance on a specific objective metric, such as accuracy, F1 score, or loss. Hyperparameters are configuration settings that control the training process, such as learning rate, batch size, and the number of layers in a neural network. The learning rate in our case ranged **beteen 0.001 and 0.1**, the **batch size values are 32, 64, and 128**. We were aiming to maximise accuracy.

**Hyperparameter Tuning Steps:**<br>
The notebook highlights hyperparameter tuning to optimize model performance by fine-tuning a pre-trained model in SageMaker.<br>
**Hyperparameter Setup:** Defines a range of hyperparameters (lr, batch-size, epochs) using SageMaker’s ContinuousParameter, CategoricalParameter, and IntegerParameter.<br>
**Hyperparameter Tuner Creation:** Creates a HyperparameterTuner object with configurations like objective_metric_name, tuning ranges, job limits, and metric definitions.<br>
**Hyperparameter Tuning Execution:** Runs the tuning process with tuner.fit using S3 paths for training and validation datasets.<br>
**Accessing Results:** Demonstrates fetching the best hyperparameters from the tuning job.<br>
**Best Hyperparameter Selection:** Constructs a dictionary of the best hyperparameters, including batch size and learning rate.<br>
**Optional Estimator Attachments:** Provides flexibility to attach and control the estimator outside the tuner context.<br>
**Metric Analysis:** Includes definitions and regex for tracking metrics like epoch_accuracy and epoch_loss during tuning.<br>
**Parallel Jobs:** Configures parallel jobs to accelerate hyperparameter tuning without exceeding resource limits.<br>
**Validation Strategy:** Specifies a split between training and validation datasets for robust tuning outcomes.<br>

<p align="center">
  <img src="./screenshots/screenshot_project_10.JPG" />
</p>
<p align="center">
  <img src="./screenshots/screenshot_project_4.JPG" />
</p>
The image above shows the completed hyperparameter jobs. The image below shows the completed training jobs of one hyperparameter job.
<p align="center">
  <img src="./screenshots/screenshot_project_5.JPG" />
</p>

<p align="center">
  <img src="./screenshots/screenshot_project_8.JPG" />
</p>

The image below shows the retrieve of the completed jobs and the job with the best result.
<p align="center">
  <img src="./screenshots/screenshot_project_16.JPG" />
</p>
<p align="center">
  <img src="./screenshots/screenshot_project_17.JPG" />
</p>

## Debugging and Profiling

Debugging and profiling in Amazon SageMaker are essential for identifying performance bottlenecks and ensuring model correctness during training and inference. SageMaker provides powerful tools like SageMaker Debugger and SageMaker Profiler to streamline these processes.

For debugging and profiling the model we defined rules, collection_configs, hook_config, and profiler_config, which we passed to the Sagemaker estimator.

### Results

The complete [profiler report](./ProfilerReport/profiler-output/profiler-report.html) can be found in the directory "ProfilerReport/project-output".

#### SageMaker Debugger Profiling Report

SageMaker Debugger auto generated a report of the last training job:
```
# Parameters
processing_job_arn = "arn:aws:sagemaker:us-east-1:861747698849:processing-job/final-training-2025-01-22--ProfilerReport-cd6d7e20"
```
You can generate similar reports on all supported training jobs. The report provides summary of training job, system resource usage statistics, framework metrics, rules summary, and detailed analysis from each rule.

#### Training job summary

The following table gives a summary about the training job. The table includes information about when the training job started and ended, how much time initialization, training loop and finalization took. Your training job started on 01/22/2025 at 12:20:05 and ran for 6534 seconds. Your training job started on 01/22/2025 at 12:20:05 and ran for 6534 seconds.. No step information was profiled from your training


<p align="center">
  <img src="./screenshots/profiler_10.JPG" />
</p>

#### System usage statistics 

The following table shows statistics of resource utilization per worker (node), such as the total CPU and GPU utilization, and the memory utilization on CPU and GPU. The table also includes the total I/O wait time and the total amount of data sent or received in bytes. The table shows min and max values as well as p99, p90 and p50 percentiles.

<p>
  <img src="./screenshots/profiler_00.JPG" />
</p>

The 95th percentile of the total GPU utilization on node algo-1 is only 0%. However, the 95th percentile of the total CPU utilization is 72%.

The following screenshot shows high CPU utilization during the training process, since we were only using CPU:

<p align="center">
  <img src="./screenshots/screenshot_project_14.JPG" />
</p>

#### Batch size 
The BatchSize rule helps to detect if GPU is underutilized because of the batch size being too small. To detect this the rule analyzes the GPU memory footprint, CPU and GPU utilization. The rule checked if the 95th percentile of CPU utilization is below cpu_threshold_p95 of 70%, the 95th percentile of GPU utilization is below gpu_threshold_p95 of 70% and the 95th percentile of memory footprint below gpu_memory_threshold_p95 of 70%. In your training job this happened 14 times. The rule skipped the first 1000 datapoints. The rule computed the percentiles over window size of 500 continuous datapoints. The rule analysed 13068 datapoints and triggered 14 times.

The following boxplots are a snapshot from the timestamps. They the total CPU utilization, the GPU utilization, and the GPU memory usage per GPU (without outliers).

<p align="center">
  <img src="./screenshots/profiler_2.JPG" />
</p>

It appears that our training job is **underutilizing** the instance. We may want to consider either **switch to a smaller instance type or to increase the batch size**. 

## Model Deployment

For this experiment whe chose the Resnet-50 (residual neural network). ResNet50 is a deep convolutional neural network (CNN) architecture that was developed by Microsoft Research in 2015. It is a variant of the popular ResNet architecture, which stands for “Residual Network.” The “50” in the name refers to the number of layers in the network, which is 50 layers deep. (Source: [medium.com](https://medium.com/@nitishkundu1993/exploring-resnet50-an-in-depth-look-at-the-model-architecture-and-code-implementation-d8d8fa67e46f))

**Model Deployment and Testing Steps:**
**Training Job Monitoring:** Fetches details of the latest training job for review, including the job name, client, and description.<br>
**Creating/Managing/Listing Endpoint Deployment:** Deploys the trained model to a SageMaker endpoint with specific instance configurations and an endpoint name. Provides commands to list active SageMaker endpoints for monitoring.<br>

**Error Logging:** Includes methods to locate inference errors or failures in S3 logs for debugging.<br>

<p align="center">
  <img src="./screenshots/screenshot_project_11.JPG" />
</p>
<p align="center">
  <img src="./screenshots/screenshot_project_12.JPG" />
</p>

**Endpoint Cleanup:** Remember to delete endpoints once testing or deployment is complete to avoid incurring costs. This can be done via console or Jupyter notebook.<br>
<p align="center">
  <img src="./screenshots/screenshot_project_15.JPG" />
</p>

