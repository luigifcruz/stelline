FROM stelline-base

ARG DEBIAN_FRONTEND=noninteractive

#
# Socket Bit Dependencies
#

RUN apt update && \
    apt install --no-install-recommends -y \
        libzmq3-dev \
        && rm -rf /var/lib/apt/lists/*

#
# CyberEther Dependencies
#

RUN apt update && \
    apt install --no-install-recommends -y \
        libxkbcommon-dev \
        spirv-cross glslang-tools libglfw3-dev \
        libvulkan-dev vulkan-validationlayers \
        python3-yaml \
        libgstreamer1.0-dev gstreamer1.0-libav \
        gstreamer1.0-plugins-base libgstreamer-plugins-bad1.0-dev \
        libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev \
        gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
        libsoapysdr-dev soapysdr-module-rtlsdr soapysdr-module-uhd \
        soapysdr-module-airspy soapysdr-module-lms7 soapysdr-tools \
        && rm -rf /var/lib/apt/lists/*

#
# BLADE Dependencies
#
# Omit libhdf5-dev as it overules the selective install from stelline-base
#  reverting to 1.10.x

RUN apt update && \
    apt install --no-install-recommends -y \
        liberfa-dev \
        libbenchmark-dev libgtest-dev \
        && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install numpy astropy pandas

#
# Workspace Setup
#

WORKDIR /workspace

#
# Clone, Build & Install filterbankc99, to support FBH5 writing
#

RUN git clone https://github.com/MydonSolutions/filterbankc99.git
RUN cd filterbankc99 \
    && git checkout v1.5.0 \
    && git submodule update --init --recursive \
    && CC=gcc-11 meson setup -Dbuildtype=debugoptimized build \
    && cd build \
    && ninja install

#
# Clone, Build & Install CyberEther
#

RUN git clone https://github.com/luigifcruz/CyberEther.git
RUN cd CyberEther \
    && git checkout v1.0.0-alpha6 \
    && rm -fr build \
    && CC=gcc-11 CXX=g++-11 meson setup -Dbuildtype=debugoptimized build \
    && cd build \
    && ninja \
    && ninja install

#
# Clone, Build & Install BLADE
#

ENV NVCC_PREPEND_FLAGS='-ccbin g++-11'
RUN git clone https://github.com/luigifcruz/blade.git
RUN cd blade \
    && git checkout v1.1 \
    && git submodule update --init --recursive \
    && rm -fr build \
    && CC=gcc-11 CXX=g++-11 meson setup -Dbuildtype=debugoptimized build \
    && cd build \
    && ninja \
    && ninja install
