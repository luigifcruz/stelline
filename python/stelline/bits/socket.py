"""
Socket bits for Stelline pipeline
"""

from typing import Any, Tuple

from holoscan.core import Application, Resource
from ..operators import ZmqTransmitterOp
from ..utils import logger


def SocketBit(app: Application, pool: Any, id: int, config: str) -> Tuple[Any, Any]:
    """
    Create a socket bit for network communication.

    Parameters
    ----------
    app : holoscan.Application
        The Holoscan application instance
    pool : holoscan.Resource
        Memory pool resource
    id : int
        Unique identifier for this bit instance
    config : str
        Configuration key to fetch parameters from

    Returns
    -------
    Tuple[Any, Any]
        (input_operator, output_operator) tuple for connecting flows
    """
    cfg = app.kwargs(config)
    address = cfg["address"]
    bind_mode = cfg.get("bind_mode", False)

    logger.info("Socket Configuration:")
    logger.info(f"  Address: {address}")
    logger.info(f"  Bind Mode: {bind_mode}")

    # Create ZMQ transmitter operator
    logger.info("Creating ZeroMQ Transmitter operator.")
    zmq_name = f"zmq-transmitter-{id}"
    zmq_op = ZmqTransmitterOp(
        fragment=app,
        name=zmq_name,
        address=address,
    )

    return (zmq_op, zmq_op)
