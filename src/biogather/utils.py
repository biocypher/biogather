"""Utility functions for biogather."""

def is_scispacy_available() -> bool:
    """Check if scispacy is available.
    
    Returns:
        bool: True if scispacy is available, False otherwise.
    """
    try:
        import scispacy  # noqa: F401
        return True
    except ImportError:
        return False 