# Installation

The recommended way to develop or run a Stelline pipeline is through a Docker container. Although Stelline has only a few dependencies, Holoscan has many. We provide a Docker container based on the official Holoscan image with the additional dependencies required for Stelline.

## 1. Clone this repository

```
$ git clone https://github.com/luigifcruz/stelline
$ cd stelline
```

## 2. Build base container

```
$ docker build -t stelline-base -f docker/Dockerfile-base .
```

## 3. Build demo container

```
$ docker build -t stelline-dev -f docker/Dockerfile-dev .
```

## 4. Run the demo container

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
    -v /mnt/raid:/mnt/raid \
    -v .:/workspace/stelline \
    -v $nvidia_icd_json:$nvidia_icd_json:ro \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
    -e DISPLAY=$DISPLAY \
    stelline
```

## 5. Fun!

This will create a Jupyter Notebook with everything pre-installed for you. All you have to do is to navigate to `http://[SPARK-IP]:8888` and start exploring. Execute `stelline -h` in the command line for more information.
