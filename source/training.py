import boto3
import sagemaker
from sagemaker import get_execution_role
import json
import tarfile
import os
from time import gmtime, strftime

def train_my_xgboost(train, code_files, script, hyperparameters={}, role=None, prefix=None, bucket=None, train_instance_type='ml.m5.xlarge'):
    
    # 创建tar.gz文件
    def create_tar_file(source_files, target=None):
        if target:
            filename = target
        else:
            _, filename = tempfile.mkstemp()

        with tarfile.open(filename, mode="w:gz") as t:
            for sf in source_files:
                # Add all files from the directory into the root of the directory structure of the tar
                t.add(sf, arcname=os.path.basename(sf))
        return filename
    # 超参数encode成json
    def json_encode_hyperparameters(hyperparameters):
        return {str(k): json.dumps(v) for (k, v) in hyperparameters.items()}
    
    
    sagemaker_session = sagemaker.session.Session()
    
    # 取得默认的bucket
    if not bucket:
        print('Using default bucket ', end='')
        bucket = sagemaker_session.default_bucket()
        print(bucket)
    
    if not code_files[0].startswith('s3://'):
        print('Uploading code to S3:', end='')
        # 把代码文件打爆
        create_tar_file(code_files, "sourcedir.tar.gz")
        # 上传代码文件
        sources = sagemaker_session.upload_data("sourcedir.tar.gz", bucket, prefix + "/code")
        print(sources)
    else:
        sources = code_files
    
    # 把代码的s3位置放进超参数
    hyperparameters['sagemaker_submit_directory']= sources
    
    # encode超参数
    hyperparameters = json_encode_hyperparameters(
        hyperparameters
    )
    
    if not role:
        print('Getting default Role', end='')
        role = get_execution_role()
        print(role)
    
    # 放入如下内容
    # docker ecr链接
    # role
    # 同时训练的数量
    # 机器类型
    # training jobs 前缀
    # 超参数
    est = sagemaker.estimator.Estimator(
        '337058716437.dkr.ecr.ca-central-1.amazonaws.com/xgboost_001',
        role,
        train_instance_count=1,
        train_instance_type=train_instance_type,
        # train_instance_type="local",
        base_job_name=prefix,
        hyperparameters=hyperparameters,
    )
    
    # 这个可以做映射的文件，假如有666，那么文件会被挂载到/opt/ml/input/data/666/
    est.fit({"train": train})
    
    
train = 's3://ca-central-sagemaker-test/iris-data/'
script = 'train.py'

role = 'arn:aws:iam::337058716437:role/SageMaker-Execution'
code_files = ["code/train.py"]
script = 'train.py'
prefix = 'test-mlops'

hyperparameters = {
                     "sagemaker_program": script,
                     "hp1": {'xgboost':'123',
                             'test':'ttt'
                            },
                     "hp2": 300,
                     "hp3": 0.001,
                   }
train_my_xgboost(train, code_files, script, hyperparameters=hyperparameters,
                 role=role,
                 prefix=prefix
                )

print("Training Finished!!")




#################
#################
#################
# Deployment


'''
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

'''