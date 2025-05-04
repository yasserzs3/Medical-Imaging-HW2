from typing import Optional, Dict, Any
from .ssd_lite import SSDLiteWrapper

def load_model(name: str, num_classes: int, **kwargs) -> Any:
    """
    Load an SSD model by name
    
    Args:
        name (str): Model name ('ssd_lite')
        num_classes (int): Number of classes
        **kwargs: Additional model-specific arguments
        
    Returns:
        Any: Loaded model
    """
    if name == 'ssd_lite':
        pretrained = kwargs.get('pretrained', True)
        model_wrapper = SSDLiteWrapper(num_classes=num_classes, pretrained=pretrained)
        return model_wrapper.get_model()
    else:
        raise ValueError(f"Unknown model name: {name}") 