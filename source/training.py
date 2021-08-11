import boto3
import sagemaker
from sagemaker import get_execution_role
import json
import tarfile
import os

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
    
    
train = 's3://sagemaker-ca-central-1-337058716437/script-mode-container-2/train/'
script = 'train.py'

role = 'arn:aws:iam::337058716437:role/SageMaker-Execution'
code_files = ["code/source_dir/train.py"]
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

print("Training Finished!")