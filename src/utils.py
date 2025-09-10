import yaml

def load_config_basic(config_path: str):
    """Load YAML config using basic PyYAML"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
