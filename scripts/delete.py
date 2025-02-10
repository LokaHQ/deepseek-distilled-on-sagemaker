"""Script for deleting the model from AWS SageMaker"""

import argparse

from dotenv import load_dotenv
from loguru import logger
from sagemaker.huggingface import HuggingFacePredictor

load_dotenv()


def delete_model(endpoint_name):
    """
    Delete the SageMaker endpoint

    Args:
        endpoint_name (str): Name of the SageMaker endpoint to delete

    Returns:
        None

    Raises:
        Exception: If the endpoint deletion fails
    """
    try:
        logger.info(f"Deleting endpoint: {endpoint_name}")
        predictor = HuggingFacePredictor(
            endpoint_name=args.endpoint_name
        )
        predictor.delete_endpoint(delete_endpoint_config=True)
        logger.info(f"Endpoint {endpoint_name} deleted successfully")
    except Exception as e:
        logger.error(f"Failed to delete endpoint {endpoint_name}: {e}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete SageMaker Model (Endpoint) script"
    )
    parser.add_argument(
        "--endpoint_name",
        required=True,
        help="SageMaker endpoint name to delete",
    )

    args = parser.parse_args()

    delete_model(endpoint_name=args.endpoint_name)
