"""Script for deployment of the model to AWS SageMaker"""

import argparse
import json
import os

from dotenv import load_dotenv
from loguru import logger
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

load_dotenv()


def deploy_model(
    hf_model_id: str,
    role: str,
    region: str,
    instance_type: str,
    initial_instance_count: int,
    container_startup_health_check_timeout: int,
    sm_num_gpus: int,
) -> None:
    """
    Deploy a HuggingFace model to SageMaker.

    Args:
        hf_model_id (str): The HuggingFace model ID.
        role (str): The ARN of the SageMaker execution role.
        region (str): The AWS region.
        instance_type (str): The instance type for deployment.
        initial_instance_count (int): The initial number of instances.
        container_startup_health_check_timeout (int): The container startup health check timeout in seconds.
        sm_num_gpus (int): The number of GPUs to use.
    """
    logger.info("Starting deployment process...")

    model_name = hf_model_id.split("/")[-1].lower().replace(".", "-")
    hub = {"HF_MODEL_ID": hf_model_id, "SM_NUM_GPUS": json.dumps(sm_num_gpus)}
    huggingface_model = HuggingFaceModel(
        image_uri=get_huggingface_llm_image_uri(
            "huggingface", version="3.0.1", region=region
        ),
        env=hub,
        role=role,
        name=model_name,
    )
    endpoint_name = f"{model_name}-ep"
    huggingface_model.deploy(
        endpoint_name=endpoint_name,
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        container_startup_health_check_timeout=container_startup_health_check_timeout,
    )
    logger.info(f"Model deployed at endpoint: {endpoint_name}")
    logger.info("Deployment process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model_id",
        default=os.getenv("hf_model_id"),
        help="The HuggingFace model ID to deploy.",
    )
    parser.add_argument(
        "--role_arn",
        default=os.getenv("role_arn"),
        help="The ARN of the SageMaker execution role.",
    )
    parser.add_argument(
        "--region_info",
        default=os.getenv("region_info"),
        help="The AWS region where the model will be deployed.",
    )
    parser.add_argument(
        "--instance_type",
        default=os.getenv("instance_type"),
        help="The instance type for deployment.",
    )
    parser.add_argument(
        "--initial_instance_count",
        type=int,
        default=int(os.getenv("initial_instance_count")),
        help="The initial number of instances for deployment.",
    )
    parser.add_argument(
        "--container_startup_health_check_timeout",
        type=int,
        default=int(os.getenv("container_startup_health_check_timeout")),
        help="The container startup health check timeout in seconds.",
    )
    parser.add_argument(
        "--sm_num_gpus",
        type=int,
        default=int(os.getenv("sm_num_gpus")),
        help="The number of GPUs to use for the deployment.",
    )
    args = parser.parse_args()

    deploy_model(
        hf_model_id=args.hf_model_id,
        role=args.role_arn,
        region=args.region_info,
        instance_type=args.instance_type,
        initial_instance_count=args.initial_instance_count,
        container_startup_health_check_timeout=args.container_startup_health_check_timeout,
        sm_num_gpus=args.sm_num_gpus,
    )
