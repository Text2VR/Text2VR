"""
Custom exceptions for Text2VR application
"""

class Text2VRException(Exception):
    """Base exception for Text2VR"""
    pass

class ConfigurationError(Text2VRException):
    """Configuration related errors"""
    pass

class DockerServiceError(Text2VRException):
    """Docker service related errors"""
    pass

class WorkflowError(Text2VRException):
    """Workflow execution errors"""
    pass

class ResourceNotFoundError(Text2VRException):
    """Resource not found errors"""
    pass
