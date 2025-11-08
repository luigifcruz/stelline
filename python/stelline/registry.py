"""
Bit registry system for Stelline pipeline components.

This module provides a decorator-based registry system for registering
and creating bit instances dynamically.
"""

from typing import Any, Callable, Dict, List, Tuple

# Global registry to store bit types and their corresponding functions
_bit_registry: Dict[str, Callable] = {}


def register_bit(bit_name: str):
    """
    Decorator to register a bit function with a given name.

    Parameters
    ----------
    bit_name : str
        The string identifier for the bit type

    Returns
    -------
    Callable
        The decorator function

    Example
    -------
    @register_bit("socket_bit")
    def SocketBit(app, pool, id, config):
        # Implementation here
        pass
    """

    def decorator(bit_func: Callable) -> Callable:
        _bit_registry[bit_name] = bit_func
        return bit_func

    return decorator


def get_bit_function(bit_name: str) -> Callable:
    """
    Get a bit function by its registered name.

    Parameters
    ----------
    bit_name : str
        The string identifier for the bit type

    Returns
    -------
    Callable
        The bit function

    Raises
    ------
    ValueError
        If the bit name is not found in the registry
    """
    if bit_name not in _bit_registry:
        available_bits = list(_bit_registry.keys())
        raise ValueError(f"Unknown bit: '{bit_name}'. Available bits: {available_bits}")
    return _bit_registry[bit_name]


def create_bit(
    bit_name: str, app: Any, pool: Any, idx: int, bit_config_key: str
) -> Tuple[Any, Any]:
    """
    Create a bit instance by name using the registry.

    Parameters
    ----------
    bit_name : str
        The string identifier for the bit type
    app : holoscan.Application
        The Holoscan application instance
    pool : holoscan.Resource
        Memory pool resource
    idx : int
        Unique identifier for this bit instance
    bit_config_key : str
        Configuration key to fetch parameters from

    Returns
    -------
    Tuple[Any, Any]
        (input_operator, output_operator) tuple for connecting flows
    """
    bit_func = get_bit_function(bit_name)
    return bit_func(app, pool, idx, bit_config_key)


def register_existing_bit(bit_name: str, bit_func: Callable) -> None:
    """
    Register an existing bit function with the registry.

    This is useful for registering bits that weren't decorated with @register_bit.

    Parameters
    ----------
    bit_name : str
        The string identifier for the bit type
    bit_func : Callable
        The bit function to register
    """
    _bit_registry[bit_name] = bit_func


def list_bits() -> List[str]:
    """
    Get a list of all registered bit names.

    Returns
    -------
    List[str]
        List of registered bit names
    """
    return list(_bit_registry.keys())


def clear_registry() -> None:
    """
    Clear all registered bits from the registry.

    This is mainly useful for testing purposes.
    """
    _bit_registry.clear()
