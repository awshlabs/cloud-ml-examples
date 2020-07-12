# <div align="left"> RAPIDS & SageMaker Machine Learning Services Integration Lab</div>

RAPIDS is a suite of open-source libraries that bring GPU acceleration
to data science pipelines. Users building cloud-based hyperparameter
optimization experiments can take advantage of this acceleration
throughout their workloads to build models faster, cheaper, and more
easily on the cloud platform of their choice.

This repository provides example notebooks and "getting started" code
samples to help you integrate RAPIDS with the hyperparameter
optimization services on AWS Sagemaker.  A step-by-step guide to
launch an example hyperparameter optimization job.

The example job will use RAPIDS
[cuDF](https://github.com/rapidsai/cudf) to load and preprocess 
millions of rows of airline arrival and departure data and build a model
to predict whether or not a flight will arrive on time. It
demonstrates both [cuML](https://github.com/rapidsai/cuml) Random
Forests and GPU-accelerated XGBoost modeling.

## Setting up SageMaker notebook

1. Create a SageMaker Notebook Instance in us-east-1. (N. Virginia).
1. Select ml.t3.2xlarge instance type.
1. Under **Additional configuration**, Set volume size to be 50 GB.
1. Create a new IAM role with Any S3 bucket access. 
1. Click on Create Notebook instance. 
1. Once the Notebook is created, select open JupyterLab.
1. Open a terminal window:  git clone https://github.com/awshlabs/cloud-ml-examples

## AWS SageMaker
[Amazon SageMaker Step-by-step.](https://github.com/rapidsai/cloud-ml-examples/blob/master/aws/README.md "SageMaker Deployment Guide")

