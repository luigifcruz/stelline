# Stelline
This repository contains all the parts of the next-generation data processing pipeline of the Allen Telescope Array.

## Build Development Image

### 1. Clone this repository
```
$ git clone https://github.com/luigifcruz/stelline
$ cd stelline
```

### 2. Build base container
```
$ docker build -t stelline-base -f docker/Dockerfile-base .
```

### 3. Build demo container
```
$ docker build -t stelline -f docker/Dockerfile .
```

### 4. Run the demo container
```
$ nvidia_icd_json=$(find /usr/share /etc -path '*/vulkan/icd.d/nvidia_icd.json' -type f -print -quit 2>/dev/null | grep .) || (echo "nvidia_icd.json not found" >&2 && false)
$ sudo docker run -it --rm -u root \
    --net host \
    --privileged \
    --gpus=all \
    --cap-add CAP_SYS_PTRACE \
    --ipc=host \
    --volume /run/udev:/run/udev:ro \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --device=/dev/nvidia-fs* \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/huge:/mnt/huge \
    -v .:/workspace/stelline \
    -v $nvidia_icd_json:$nvidia_icd_json:ro \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
    -e DISPLAY=$DISPLAY \
    stelline
```

### 5. Compile
Once inside the container, compile the library and the applications:
```
$ cd stelline
$ meson -Dbuildtype=debugoptimized build
$ cd build
$ ninja
```

### 6. Fun!
Check each application README for further instructions. Or start developing your own application using the library.
