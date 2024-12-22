import os
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM

def log_config(log_level):
    """Configure logging for the module."""
    valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    if log_level not in valid_log_levels:
        log_level = "INFO"

    logging.basicConfig(
        level=log_level, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging level set to: {log_level}")
    
    return logger


def download_model(model_name, model_dir, hf_token, logger):
    """ 
    Downloads the tokenizer and model artifacts from HuggingFace.
    Args:
        model_name: (str) : Name of the model
        model_dir: (str) : Directory to save the model artifacts
        hf_token: (str) : Huggingface authentication token
    """
    try:
        logger.info(f"Downloading model artifacts from huggingface for: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_auth_token=hf_token
        )
        tokenizer.save_pretrained(model_dir)
        logger.info(f"Tokenizer is downloaded at: {model_dir}")
    
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            use_auth_token=hf_token
        )
        model.save_pretrained(model_dir)
        logger.info(f"Model downloaded at: {model_dir}")
    
    except Exception as e:
        logger.error(f"Error downloading model artifacts: {e}")
        raise

def main():
    """
    Main execution logic
    """

    model_name = os.getenv("MODEL_NAME", "NickyNicky/experimental-Mistral-1b-V00")
    model_dir = os.getenv("MODEL_DIR", "models/mistral-1b")
    hf_token = os.getenv("HF_TOKEN")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    print(log_level)
    # configure logging
    logger = log_config(log_level)

    # validate hugging face token
    if not hf_token:
        logger.error(f"Hugging Face token not provided. Set it using environment variable HF_TOKEN")
        raise EnvironmentError(" Missing Hugging Face token")

    # create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # download model artifacts
    try:
        download_model(model_name, model_dir, hf_token, logger)
        logger.info(f"Model '{model_name}' downloaded at: {model_dir} successfully")
    except Exception as e:
        logger.critical(f"Error occured during model download: {e}")
        raise

if __name__ == "__main__":
    main()