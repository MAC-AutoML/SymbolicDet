"""Custom exceptions for the SR system"""

class SRException(Exception):
    """Base exception class for Symbolic Regression"""
    pass

class ExpressionError(SRException):
    """Base class for expression related errors"""
    pass

class ExpressionParseError(ExpressionError):
    """Error in parsing expressions"""
    pass

class ExpressionEvalError(ExpressionError):
    """Error in evaluating expressions"""
    pass

class ConfigError(SRException):
    """Base class for configuration errors"""
    pass

class DataError(SRException):
    """Base class for data related errors"""
    pass

class ProcessError(SRException):
    """Base class for process related errors"""
    pass

class ResourceError(SRException):
    """Base class for resource management errors"""
    pass 

class EvaluationError(SRException):
    """Base class for evaluation errors"""
    pass