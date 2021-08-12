import boto3
import sagemaker
from sagemaker import get_execution_role
import json
import tarfile
import os
from time import gmtime, strftime


print('***********************\n' * 3)
print('Deploying!!!')
#endpoint名字
endpoint_name = 'test-mlops2' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
#模型列表，注意最后结尾带上 /
model_url = 's3://sagemaker-ca-central-1-337058716437/test-mlops-2021-08-12-06-13-33-859/output/'
container = '337058716437.dkr.ecr.ca-central-1.amazonaws.com/xgboost-multi'
role = 'arn:aws:iam::337058716437:role/SageMaker-Execution'

sm_client = boto3.client(service_name='sagemaker')
runtime_sm_client = boto3.client(service_name='sagemaker-runtime')

container = {
    'Image': container,
    'ModelDataUrl': model_url,
    'Mode': 'MultiModel'
}

model_name = endpoint_name

create_model_response = sm_client.create_model(
    ModelName = model_name,
    ExecutionRoleArn = role,
    Containers = [container])

endpoint_config_name = model_name
print('Endpoint config name: ' + endpoint_config_name)

create_endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType': 'ml.m5.large',
        'InitialInstanceCount': 2,
        'InitialVariantWeight': 1,
        'ModelName': model_name,
        'VariantName': 'AllTraffic'}])

print('Endpoint name: ' + endpoint_name)

create_endpoint_response = sm_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name)

print('Endpoint Arn: ' + create_endpoint_response['EndpointArn'])


resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
status = resp['EndpointStatus']
print("Endpoint Status: " + status)

print('Waiting for {} endpoint to be in service...'.format(endpoint_name))
waiter = sm_client.get_waiter('endpoint_in_service')
waiter.wait(EndpointName=endpoint_name)

print('Deployment finished! Endpoint name:{}'.format(endpoint_name))