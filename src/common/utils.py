import os
from dotenv import load_dotenv

load_dotenv()

def get_env_variable(key):
    """Get the environment variable or return exception."""
    value = os.getenv(key)
    if value is None:
        error_msg = f"Missing ENV: {key}"
        raise ValueError(error_msg)
    return value

