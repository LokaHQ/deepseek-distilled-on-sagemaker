"""Inference script for the AWS SageMaker model"""

import argparse

from dotenv import load_dotenv
from loguru import logger
from sagemaker.huggingface import HuggingFacePredictor

load_dotenv()


def generate(
    predictor: HuggingFacePredictor,
    prompt: str,
    temperature=0.7,
    max_new_tokens=128,
    top_k=50,
    top_p=0.95,
) -> dict:
    """
    Generate response using the SageMaker model

    Args:
        predictor (HuggingFacePredictor): SageMaker predictor
        prompt (str): Prompt for the model
        temperature (float): Controls randomness in generation (0.0-1.0)
        max_new_tokens (int): Maximum number of tokens to generate
        top_k (int): Top-K sampling parameter
        top_p (float): Nucleus sampling parameter (0.0-1.0)

    Returns:
        dict: Response from the model
    """
    logger.info("Generating response using the model")

    response = predictor.predict(
        {
            "inputs": prompt,
            "parameters": {
                "do_sample": True,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            },
        }
    )

    logger.info("Response generated successfully")
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for the SageMaker model"
    )
    parser.add_argument(
        "--endpoint_name",
        default="deepseek-r1-distill-qwen-1-5b-ep",
        help="SageMaker endpoint name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the meaning of life, the universe, and everything?",
        help="Prompt for the model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Controls randomness in generation (0.0-1.0)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Top-K sampling parameter"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Nucleus sampling parameter (0.0-1.0)"
    )

    args = parser.parse_args()

    predictor = HuggingFacePredictor(
        endpoint_name=args.endpoint_name
    )

    response = generate(
        predictor=predictor,
        prompt=args.prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    logger.info(f"Generated text: {response[0]['generated_text']}")
