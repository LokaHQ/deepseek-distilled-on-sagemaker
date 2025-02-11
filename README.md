# deepseek-distilled-on-sagemaker 📦

A code repository for deploying distilled versions of DeepSeek-R1 on SageMaker as a Endpoint (or any other Open Source Model)

## 📋 Prerequisites

1. **🔑 AWS Account**: Ensure you have an AWS account with the necessary permissions.
2. **🔐 IAM Role**: Create an IAM role with sufficient permissions to access SageMaker and other necessary services.
3. **SageMaker Domain**: Ensure you have a SageMaker Domain configured.
4. **Sufficient Quota**: Ensure you have sufficient quota in SageMaker, especially for the instance types you plan to use.
5. **[uv](https://docs.astral.sh/uv/)**: An extremely fast Python package and project manager for the code approach.
6. **AWS CLI**: Ensure your AWS account is configured locally (access key and secret key or use `aws-vault`).

## 🏗️ Architecture

![arch](https://github.com/user-attachments/assets/a8c999fa-8cac-4063-bedd-b5b49515c2c0)


## 🚀 Deployment Methods

### 👨🏻‍💻 Code Approach with SageMaker SDK

This method involves using code to deploy the DeepSeek models on AWS SageMaker.

#### Environment Setup

1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    ```

2. **Install `uv`**:
    ```bash
    pip install uv
    ```

3. **Sync the Environment**:
    ```bash
    uv sync
    ```

4. **Configure AWS CLI or AWS Vault**:
    - Ensure your AWS account is configured locally using AWS CLI or AWS Vault.
    ```bash
    aws configure
    ```
    - Alternatively, you can use AWS Vault for managing your credentials securely.
    ```bash
    aws-vault add <profile-name>
    ```

4. **Configure Environment Variables**:
    - Copy the example environment file and update it with your specific values.
    ```bash
    cp .env.example .env
    ```
    - Edit the `.env` file with your preferred text editor and fill in the required values:
    ```bash
    hf_model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    role_arn = 'arn:aws:iam::[account_id]:role/service-role/[AmazonSageMaker-ExecutionRole-xxxxxxxxxxxxxxx]'
    region_info = 'us-west-2'
    instance_type = 'ml.g6.2xlarge'
    initial_instance_count = 1
    container_startup_health_check_timeout = 600
    sm_num_gpus = 1
    ```

#### Deployment Steps & Testing

1. **Deploy the Model**:
    - Run the `./scripts/deploy.py` script to deploy the model to SageMaker.
    ```bash
    uv run ./scripts/deploy.py --hf_model_id <HF-MODEL-ID> --role_arn <IAM-ROLE-ARN> --region_info <AWS-REGION> --instance_type <INSTANCE-TYPE> --initial_instance_count <INSTANCE-COUNT> --container_startup_health_check_timeout <TIMEOUT> --sm_num_gpus <NUM-GPUS>
    ```

2. **Run Inference**:
    - After the model is deployed, you can run inference using the `./scripts/inference.py` script.
    ```bash
    uv run ./scripts/inference.py --endpoint_name <ENDPOINT-NAME> --prompt "<PROMPT>" --temperature <TEMPERATURE> --max_new_tokens <MAX-NEW-TOKENS> --top_k <TOP-K> --top_p <TOP-P>
    ```

3. **Run Benchmarking**:
    - You can also run benchmarking using the `./benchmark/benchmark.py` script.
    ```bash
    uv run ./benchmark/benchmark.py --endpoint_name <ENDPOINT-NAME> --region_info <AWS-REGION> --output_dir <OUTPUT-DIR> --temperature <TEMPERATURE> --max_tokens <MAX-TOKENS> --top_p <TOP-P> --max_retries <MAX-RETRIES> --cold_start_loops <COLD-START-LOOPS> --stat_loops <STAT-LOOPS>
    ```

4. **Delete the Model**:
    - To delete the model from SageMaker, you can run the `./scripts/delete.py` script.
    ```bash
    uv run ./scripts/delete.py --endpoint_name <ENDPOINT-NAME>
    ```

## 📝 Notes

- Ensure that the IAM role specified in `role_arn` has the necessary permissions to access SageMaker and other required services.
- Adjust the parameters such as `instance_type`, `initial_instance_count`, and `container_startup_health_check_timeout` as needed for your specific use case.
- Make sure you delete the endpoint once you finish testing it to avoid unnecessary charges.
- To choose the correct number of GPUs per replica, refer to the following table:

| Model                                      | Instance Type   | # of GPUs per replica |
|--------------------------------------------|-----------------|-----------------------|
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B  | ml.g6.48xlarge  | 8                     |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B   | ml.g6.12xlarge  | 4                     |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B   | ml.g6.12xlarge  | 4                     |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | ml.g6.2xlarge   | 1                     |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    | ml.g6.2xlarge   | 1                     |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  | ml.g6.2xlarge   | 1                     |

## 📂 Repo Structure

```
.
├── benchmark
│   └── benchmark.py
├── scripts
│   ├── delete.py
│   ├── deploy.py
│   └── inference.py
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
