"""
Transport bits for Stelline pipeline
"""

from typing import Any, Tuple

from holoscan import conditions
from holoscan.core import Application, Resource

from ..operators import (
    AtaReceiverOp,
    DummyReceiverOp,
    SorterOp,
)
from ..registry import register_bit
from ..types import BlockShape
from ..utils import logger


@register_bit("transport_bit")
def TransportBit(app: Application, pool: Any, id: int, config: str) -> Tuple[Any, Any]:
    """
    Create a transport bit for receiving network data or generating test data.

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
    mode = cfg.get("mode")
    total_block = cfg.get("total_block")

    logger.info("Transport Configuration:")
    logger.info(f"  Mode: {mode}")

    total_block_shape = BlockShape(
        total_block["number_of_antennas"],
        total_block["number_of_channels"],
        total_block["number_of_samples"],
        total_block["number_of_polarizations"],
    )

    if mode == "ata":
        partial_block = cfg.get("partial_block")
        offset_block = cfg.get("offset_block")
        max_concurrent_blocks = cfg.get("max_concurrent_blocks")
        output_pool_size = cfg.get("output_pool_size")
        enable_csv_logging = cfg.get("enable_csv_logging") or False
        sorter_depth = cfg.get("sorter_depth") or 16

        # Network configuration
        gpu_device_id = cfg.get("gpu_device_id")
        interface_address = cfg.get("interface_address")
        master_core = cfg.get("master_core")
        worker_core = cfg.get("worker_core")
        max_concurrent_bursts = cfg.get("max_concurrent_bursts")
        packets_per_burst = cfg.get("packets_per_burst")

        # Packet configuration
        packet_parser_type = "ata"
        packet_header_offset = 42
        packet_header_size = 16
        packet_data_size = 6144

        # UDP configuration
        udp_src_port = 50100
        udp_dst_port = 50000

        logger.info(f"  Network:")
        logger.info(f"    GPU Device ID: {gpu_device_id}")
        logger.info(f"    Interface Address: {interface_address}")
        logger.info(f"    Master Core: {master_core}")
        logger.info(f"    Worker Core: {worker_core}")
        logger.info(f"    Packet Parser Type: {packet_parser_type}")
        logger.info(f"    Packet Header Offset: {packet_header_offset}")
        logger.info(f"    Packet Header Size: {packet_header_size}")
        logger.info(f"    Packet Data Size: {packet_data_size}")
        logger.info(f"    Packets per Burst: {packets_per_burst}")
        logger.info(f"    Max Concurrent Bursts: {max_concurrent_bursts}")

        logger.info(f"  Processing:")
        logger.info(f"    Max Concurrent Blocks: {max_concurrent_blocks}")
        logger.info(f"    Output Pool Size: {output_pool_size}")
        logger.info(f"    Enable CSV Logging: {enable_csv_logging}")
        logger.info(f"    Sorter Depth: {sorter_depth}")

        logger.info(f"  Block Configuration:")
        logger.info(f"    Total Block: {total_block}")
        logger.info(f"    Partial Block: {partial_block}")
        logger.info(f"    Offset Block: {offset_block}")

        partial_block_shape = BlockShape(
            partial_block["number_of_antennas"],
            partial_block["number_of_channels"],
            partial_block["number_of_samples"],
            partial_block["number_of_polarizations"],
        )
        offset_block_shape = BlockShape(
            offset_block["number_of_antennas"],
            offset_block["number_of_channels"],
            offset_block["number_of_samples"],
            offset_block["number_of_polarizations"],
        )

        # Create AtaReceiver operator
        logger.info("Creating ATA receiver.")
        ata_receiver_name = f"transport-ata-receiver-{id}"
        ata_receiver_op = AtaReceiverOp(
            fragment=app,
            gpu_device_id=gpu_device_id,
            interface_address=interface_address,
            master_core=master_core,
            worker_core=worker_core,
            packet_parser_type=packet_parser_type,
            packet_header_offset=packet_header_offset,
            packet_header_size=packet_header_size,
            packet_data_size=packet_data_size,
            packets_per_burst=packets_per_burst,
            max_concurrent_bursts=max_concurrent_bursts,
            max_concurrent_blocks=max_concurrent_blocks,
            udp_src_port=udp_src_port,
            udp_dst_port=udp_dst_port,
            total_block=total_block_shape,
            partial_block=partial_block_shape,
            offset_block=offset_block_shape,
            output_pool_size=output_pool_size,
            enable_csv_logging=enable_csv_logging,
            name=ata_receiver_name,
        )

        # Create Sorter operator
        sorter_name = f"transport-sorter-{id}"
        sorter_op = SorterOp(fragment=app, name=sorter_name, depth=sorter_depth)

        # Connect operators
        app.add_flow(ata_receiver_op, sorter_op, {("dsp_block_out", "dsp_block_in")})

        return (ata_receiver_op, sorter_op)

    elif mode == "dummy":
        logger.info(f"  Total Block: {total_block}")

        # Create DummyReceiver operator
        logger.info("Creating Dummy Receiver operator.")
        dummy_receiver_name = f"transport-dummy-receiver-{id}"
        dummy_receiver_op = DummyReceiverOp(
            fragment=app,
            total_block=total_block_shape,
            name=dummy_receiver_name,
        )

        return (dummy_receiver_op, dummy_receiver_op)

    else:
        raise ValueError(f"Unsupported transport mode: {mode}")
