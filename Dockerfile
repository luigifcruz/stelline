ARG BASE_IMAGE=ghcr.io/luigifcruz/cyberether:ubuntu24-x86_64-cuda
FROM ${BASE_IMAGE} AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      cmake \
      libdpdk-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /workspace
WORKDIR /workspace

RUN git config --global --add safe.directory /workspace

ARG BUILD_TYPE=release

RUN rm -rf build && \
    meson setup build \
      -Dbuildtype=${BUILD_TYPE}
RUN meson compile -C build stelline_cep

RUN mkdir -p /out && \
    cp build/stelline.cep /out/stelline.cep

FROM scratch AS artifact
COPY --from=build /out/ /
