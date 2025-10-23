"""
Transport bits for Stelline pipeline
"""

from typing import Any, Tuple

from holoscan.core import Application, Resource
from holoscan import conditions

from ..types import BlockShape
from ..utils import logger
from ..operators import (
    ReceiverOp,
    DummyReceiverOp,
    SorterOp,
    AdvNetworkOpRx,
)

from stelline.operators._transport_ops import (
    TRANSPORT_HEADER_SIZE,
    VOLTAGE_HEADER_SIZE,
    VOLTAGE_DATA_SIZE,
)


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
    mode = cfg["mode"]
    total_block = cfg["total_block"]

    logger.info("Transport Configuration:")
    logger.info(f"  Mode: {mode}")

    total_block_shape = BlockShape(
        total_block["number_of_antennas"],
        total_block["number_of_channels"],
        total_block["number_of_samples"],
        total_block["number_of_polarizations"],
    )

    if mode == "ata":
        partial_block = cfg["partial_block"]
        offset_block = cfg["offset_block"]
        concurrent_blocks = cfg["concurrent_blocks"]
        output_pool_size = cfg["output_pool_size"]
        enable_csv_logging = cfg.get("enable_csv_logging", False)
        sorter_depth = cfg.get("sorter_depth", 16)

        # RDMA configuration
        rdma_gpu = cfg["rdma_gpu"]
        rdma_nic = cfg["rdma_nic"]
        rdma_master_core = cfg["rdma_master_core"]
        rdma_worker_cores = cfg["rdma_worker_cores"]
        rdma_max_bursts = cfg["rdma_max_bursts"]
        rdma_burst_size = cfg["rdma_burst_size"]

        max_packet_size = (
            TRANSPORT_HEADER_SIZE + VOLTAGE_HEADER_SIZE + VOLTAGE_DATA_SIZE
        )
        header_data_split = TRANSPORT_HEADER_SIZE + VOLTAGE_HEADER_SIZE

        logger.info("  RDMA:")
        logger.info(f"    GPU: {rdma_gpu}")
        logger.info(f"    NIC: {rdma_nic}")
        logger.info(f"    Master Core: {rdma_master_core}")
        logger.info(f"    Worker Cores: {rdma_worker_cores}")
        logger.info(f"    Max Bursts: {rdma_max_bursts}")
        logger.info(f"    Burst Size: {rdma_burst_size}")
        logger.info(f"    Max Packet Size: {max_packet_size}")
        logger.info(f"    Header Data Boundary: {header_data_split}")
        logger.info(f"  Concurrent Blocks: {concurrent_blocks}")
        logger.info(f"  Output Pool Size: {output_pool_size}")
        logger.info(f"  Enable CSV Logging: {enable_csv_logging}")
        logger.info(f"  Total Block: {total_block}")
        logger.info(f"  Partial Block: {partial_block}")
        logger.info(f"  Offset Block: {offset_block}")
        logger.info(f"  Sorter Depth: {sorter_depth}")

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

        # Create Advanced Network Operator for RDMA
        logger.info("Creating ATA receiver with RDMA.")
        ano_rx_name = f"transport-ano-rx-{id}"
        ano_rx_op = AdvNetworkOpRx(
            fragment=app,
            master_core=rdma_master_core,
            interface_name=rdma_nic,
            gpu_device=rdma_gpu,
            worker_cores=rdma_worker_cores,
            max_packet_size=max_packet_size,
            num_concurrent_batches=rdma_max_bursts,
            batch_size=rdma_burst_size,
            header_data_split=header_data_split,
            udp_src_port=10000,
            udp_dst_port=50000,
            output_port_name="bench_rx_out",
            queue_name="adc_rx",
            flow_name="adc_rx",
            name=ano_rx_name,
        )

        # Create Receiver operator
        receiver_name = f"transport-receiver-{id}"
        receiver_op = ReceiverOp(
            fragment=app,
            name=receiver_name,
            concurrent_blocks=concurrent_blocks,
            total_block=total_block_shape,
            partial_block=partial_block_shape,
            offset_block=offset_block_shape,
            output_pool_size=output_pool_size,
            enable_csv_logging=enable_csv_logging,
        )

        # Create Sorter operator
        sorter_name = f"transport-sorter-{id}"
        sorter_op = SorterOp(fragment=app, name=sorter_name, depth=sorter_depth)

        # Connect operators
        app.add_flow(ano_rx_op, receiver_op, {("bench_rx_out", "burst_in")})
        app.add_flow(receiver_op, sorter_op, {("dsp_block_out", "dsp_block_in")})

        return (receiver_op, sorter_op)

    elif mode == "dummy":
        logger.info(f"  Total Block: {total_block}")

        # Create DummyReceiver operator
        logger.info("Creating Dummy Receiver operator.")
        dummy_receiver_name = f"transport-dummy-receiver-{id}"
        dummy_receiver_op = DummyReceiverOp(
            fragment=app,
            name=dummy_receiver_name,
            total_block=total_block_shape,
        )

        return (dummy_receiver_op, dummy_receiver_op)

    else:
        raise ValueError(f"Unsupported transport mode: {mode}")
