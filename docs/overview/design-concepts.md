# Design Concepts

- Importance of hardware acceleration.
- Structure overview (operators, bits, and recipes).

## Software Defined Observatory

- Concept
  - Placeholder for concise definition and elevator pitch.
- Three Pillars of Real-time Signal Processing
  - Transport
    - High-throughput ingest via RDMA-capable NICs (e.g., ConnectX) with GPU Direct RDMA when available.
    - NUMA-aware buffering and predictable latency under load.
  - Compute
    - GPU-accelerated DSP with BLADE operators (beamformer, correlator) and FRBNN inference path.
    - Composable Holoscan operators with consistent interfaces.
  - Storage
    - NVMe persistence with GPUDirect Storage; automatic fallback to CPU-mediated I/O when necessary.
    - Optional casting/permutation to align with downstream formats.
- Why Now
  - Placeholder for hardware and ecosystem drivers.
- Outcomes
  - Placeholder for benefits and measurable goals.
