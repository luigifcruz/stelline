from holoscan.core import Application, Resource
from ..operators import (
    DummyWriterOp,
    SimpleWriterOp,
    SimpleWriterRdmaOp,
    Fbh5WriterRdmaOp,
    Uvh5WriterRdmaOp,
)
from ..utils import logger
from typing import Tuple, Any


def FilesystemBit(
    app: Application,
    pool: Resource,
    id: int,
    config: str,
) -> Tuple[Any, Any]:
    cfg = app.kwargs(config)
    mode = cfg["mode"]
    file_path = cfg.get("file_path", "./file.bin")

    logger.info("Filesystem Configuration:")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  File Path: {file_path}")

    if mode == "simple_writer":
        logger.info("Creating Simple Writer operator.")
        writer_name = f"filesystem-simple-writer-{id}"
        writer_op = SimpleWriterOp(
            fragment=app,
            name=writer_name,
            file_path=file_path,
        )
    elif mode == "simple_writer_rdma":
        logger.info("Creating Simple Writer RDMA operator.")
        writer_name = f"filesystem-simple-writer-rdma-{id}"
        writer_op = SimpleWriterRdmaOp(
            fragment=app,
            name=writer_name,
            file_path=file_path,
        )
    elif mode == "dummy_writer":
        logger.info("Creating Dummy Writer operator.")
        writer_name = f"filesystem-dummy-writer-{id}"
        writer_op = DummyWriterOp(
            fragment=app,
            name=writer_name,
        )
    elif mode == "fbh5_writer_rdma":
        logger.info("Creating FBH5 Writer RDMA operator.")
        writer_name = f"filesystem-fbh5-writer-rdma-{id}"
        writer_op = Fbh5WriterRdmaOp(
            fragment=app,
            name=writer_name,
            file_path=file_path,
        )
    elif mode == "uvh5_writer_rdma":
        writer_name = f"filesystem-uvh5-writer-rdma-{id}"
        telinfo_file_path = cfg["telinfo_file_path"]
        obsantinfo_file_path = cfg["obsantinfo_file_path"]
        iers_file_path = cfg["iers_file_path"]
        logger.info(f"  Telinfo Path: {telinfo_file_path}")
        logger.info(f"  Obsantinfo Path: {obsantinfo_file_path}")
        logger.info(f"  IERS Path: {iers_file_path}")
        logger.info("Creating FBH5 Writer RDMA operator.")
        writer_op = Uvh5WriterRdmaOp(
            fragment=app,
            name=writer_name,
            output_file_path=file_path,
            telinfo_file_path=telinfo_file_path,
            obsantinfo_file_path=obsantinfo_file_path,
            iers_file_path=iers_file_path,
        )
    else:
        raise ValueError(f"Unsupported filesystem mode: {mode}")

    return (writer_op, writer_op)


__all__ = ["FilesystemBit"]
