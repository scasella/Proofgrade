class ProofgradeError(Exception):
    """Base runtime error for the public proof-grading package."""


class ConfigurationError(ProofgradeError):
    """Raised when configuration values are missing or invalid."""


class ProviderError(ProofgradeError):
    """Raised when the model provider cannot complete a grading request."""


class UnsupportedVariantError(ConfigurationError):
    """Raised when an unknown prompt variant is selected."""

