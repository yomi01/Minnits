import os
import json
import getpass
from pathlib import Path
from typing import Dict, Optional
import keyring
import dotenv

# Constants
APP_NAME = "ConversationRecorder"
OPENAI_KEY_NAME = "openai_api_key"
HUGGINGFACE_KEY_NAME = "huggingface_token"
OLLAMA_CONFIG_NAME = "ollama_config"

# Determine config directory based on OS
def get_config_dir() -> Path:
    """Get the appropriate configuration directory for the application"""
    config_dir = Path.home() / ".conversation_recorder"
    config_dir.mkdir(exist_ok=True)
    return config_dir

def save_api_keys(openai_key: Optional[str] = None, 
                  huggingface_token: Optional[str] = None,
                  ollama_host: Optional[str] = None,
                  ollama_model: Optional[str] = None) -> Dict[str, bool]:
    """
    Save API keys securely using keyring and create a .env file
    
    Args:
        openai_key: OpenAI API key
        huggingface_token: HuggingFace token
        ollama_host: Ollama host URL (e.g. http://localhost:11434)
        ollama_model: Ollama model name to use (e.g. gemma:2b)
    
    Returns:
        Dictionary indicating which keys/configs were saved
    """
    result = {"openai": False, "huggingface": False, "ollama": False}
    
    # Save to keyring
    if openai_key:
        keyring.set_password(APP_NAME, OPENAI_KEY_NAME, openai_key)
        result["openai"] = True
    
    if huggingface_token:
        keyring.set_password(APP_NAME, HUGGINGFACE_KEY_NAME, huggingface_token)
        result["huggingface"] = True
    
    if ollama_host or ollama_model:
        # Create or update ollama config
        current_config = {}
        try:
            saved_config = keyring.get_password(APP_NAME, OLLAMA_CONFIG_NAME)
            if saved_config:
                current_config = json.loads(saved_config)
        except:
            pass
        
        # Update with new values
        if ollama_host:
            current_config["host"] = ollama_host
        if ollama_model:
            current_config["model"] = ollama_model
            
        # Save back to keyring
        if current_config:
            keyring.set_password(APP_NAME, OLLAMA_CONFIG_NAME, json.dumps(current_config))
            result["ollama"] = True
    
    # Create a .env file in the project root
    env_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_file = env_path / ".env"
    
    env_vars = {}
    if os.path.exists(env_file):
        dotenv.load_dotenv(env_file)
        # Keep existing variables
        for key, value in os.environ.items():
            if key.startswith("OPENAI_") or key.startswith("HUGGINGFACE_") or key.startswith("OLLAMA_"):
                env_vars[key] = value
    
    # Update with new values if provided
    if openai_key:
        env_vars["OPENAI_API_KEY"] = openai_key
    
    if huggingface_token:
        env_vars["HUGGINGFACE_TOKEN"] = huggingface_token
    
    if ollama_host:
        env_vars["OLLAMA_HOST"] = ollama_host
    
    if ollama_model:
        env_vars["OLLAMA_MODEL"] = ollama_model
    
    # Write updated .env file
    with open(env_file, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    return result

def load_api_keys() -> Dict[str, Optional[str]]:
    """
    Load API keys from keyring or environment variables
    
    Returns:
        Dictionary containing the API keys and configuration
    """
    result = {
        "openai_api_key": None,
        "huggingface_token": None,
        "ollama_host": "http://localhost:11434",  # Default Ollama host
        "ollama_model": "gemma:12b"  # Default to Gemma 12B model
    }
    
    # Check environment variables first (they take precedence)
    if os.environ.get("OPENAI_API_KEY"):
        result["openai_api_key"] = os.environ.get("OPENAI_API_KEY")
    else:
        # Try to get from keyring
        try:
            result["openai_api_key"] = keyring.get_password(APP_NAME, OPENAI_KEY_NAME)
        except:
            pass
    
    if os.environ.get("HUGGINGFACE_TOKEN"):
        result["huggingface_token"] = os.environ.get("HUGGINGFACE_TOKEN")
    else:
        # Try to get from keyring
        try:
            result["huggingface_token"] = keyring.get_password(APP_NAME, HUGGINGFACE_KEY_NAME)
        except:
            pass
    
    # Load Ollama configuration
    if os.environ.get("OLLAMA_HOST"):
        result["ollama_host"] = os.environ.get("OLLAMA_HOST")
    
    if os.environ.get("OLLAMA_MODEL"):
        result["ollama_model"] = os.environ.get("OLLAMA_MODEL")
    
    # Try to get Ollama config from keyring
    try:
        ollama_config = keyring.get_password(APP_NAME, OLLAMA_CONFIG_NAME)
        if ollama_config:
            config_dict = json.loads(ollama_config)
            if "host" in config_dict:
                result["ollama_host"] = config_dict["host"]
            if "model" in config_dict:
                result["ollama_model"] = config_dict["model"]
    except:
        pass
    
    return result

def prompt_for_api_keys() -> Dict[str, Optional[str]]:
    """
    Prompt the user to enter API keys and save them
    
    Returns:
        Dictionary containing the entered API keys
    """
    print("Conversation Recorder & Summarizer - API Key Setup")
    print("=" * 50)
    print("Your API keys will be stored securely using the system's keyring.")
    print("You can skip entering a key by pressing Enter.")
    print()
    
    keys = load_api_keys()
    
    # Check if keys already exist and show masked version
    if keys["openai_api_key"]:
        masked_key = f"{keys['openai_api_key'][:4]}{'*' * (len(keys['openai_api_key']) - 8)}{keys['openai_api_key'][-4:]}"
        print(f"OpenAI API Key already exists: {masked_key}")
        update = input("Update OpenAI API Key? (y/n): ").lower() == 'y'
        if update:
            openai_key = getpass.getpass("Enter your OpenAI API Key: ")
        else:
            openai_key = keys["openai_api_key"]
    else:
        print("OpenAI API Key (optional when using Ollama):")
        openai_key = getpass.getpass("Enter your OpenAI API Key (or press Enter to skip): ")
    
    if keys["huggingface_token"]:
        masked_token = f"{keys['huggingface_token'][:4]}{'*' * (len(keys['huggingface_token']) - 8)}{keys['huggingface_token'][-4:]}"
        print(f"HuggingFace Token already exists: {masked_token}")
        update = input("Update HuggingFace Token? (y/n): ").lower() == 'y'
        if update:
            huggingface_token = getpass.getpass("Enter your HuggingFace Token (optional): ")
        else:
            huggingface_token = keys["huggingface_token"]
    else:
        print("HuggingFace Token is optional but recommended for speaker diarization.")
        huggingface_token = getpass.getpass("Enter your HuggingFace Token (optional): ")
    
    # Ollama configuration
    print("\nOllama Configuration:")
    if keys["ollama_host"]:
        print(f"Current Ollama host: {keys['ollama_host']}")
        update = input("Update Ollama host? (y/n): ").lower() == 'y'
        if update:
            ollama_host = input("Enter Ollama host URL (default: http://localhost:11434): ") or "http://localhost:11434"
        else:
            ollama_host = keys["ollama_host"]
    else:
        ollama_host = input("Enter Ollama host URL (default: http://localhost:11434): ") or "http://localhost:11434"
    
    if keys["ollama_model"]:
        print(f"Current Ollama model: {keys['ollama_model']}")
        update = input("Update Ollama model? (y/n): ").lower() == 'y'
        if update:
            ollama_model = input("Enter Ollama model name (default: gemma:12b): ") or "gemma:12b"
        else:
            ollama_model = keys["ollama_model"]
    else:
        ollama_model = input("Enter Ollama model name (default: gemma:12b): ") or "gemma:12b"
    
    # Save the keys and configuration
    save_result = save_api_keys(
        openai_key=openai_key, 
        huggingface_token=huggingface_token,
        ollama_host=ollama_host,
        ollama_model=ollama_model
    )
    
    if save_result["openai"]:
        print("✓ OpenAI API Key saved successfully")
    if save_result["huggingface"]:
        print("✓ HuggingFace Token saved successfully")
    if save_result["ollama"]:
        print("✓ Ollama configuration saved successfully")
    
    return {
        "openai_api_key": openai_key if openai_key else keys["openai_api_key"],
        "huggingface_token": huggingface_token if huggingface_token else keys["huggingface_token"],
        "ollama_host": ollama_host,
        "ollama_model": ollama_model
    }

def initialize_api_keys() -> Dict[str, Optional[str]]:
    """
    Initialize API keys - load from storage or prompt if not found
    
    Returns:
        Dictionary containing API keys
    """
    keys = load_api_keys()
    
    # If necessary keys are missing, prompt for keys
    if not keys["ollama_host"] or not keys["ollama_model"]:
        return prompt_for_api_keys()
    
    return keys

if __name__ == "__main__":
    # Test the API key management
    keys = prompt_for_api_keys()
    print("\nLoaded keys:")
    for key, value in keys.items():
        if value:
            if key in ["openai_api_key", "huggingface_token"]:
                masked = f"{value[:3]}{'*' * (len(value) - 6)}{value[-3:]}" if value else "Not set"
                print(f"- {key}: {masked}")
            else:
                print(f"- {key}: {value}")
        else:
            print(f"- {key}: Not set")