#!/bin/bash

set -e

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true

DOCKER_IMAGE="stelline-dev"
MIN_NVIDIA_VERSION="470.0"
MIN_DOCKER_VERSION="24.0"
CONFIG_FILE="/etc/stelline/config.json"
SYSTEMD_SERVICE="/etc/systemd/system/stelline.service"
CLI_TOOL="/usr/local/bin/stelline-host"

C_PRIMARY="212"
C_SUCCESS="46"
C_ERROR="196"
C_WARNING="214"
C_INFO="99"
C_WHITE="15"
C_MUTED="246"

setup_gum() {
    if command -v gum &> /dev/null; then
        GUM="gum"
        return 0
    fi

    echo "Initializing..."

    ARCH=$(uname -m)
    case $ARCH in
        x86_64) ARCH="x86_64" ;;
        aarch64|arm64) ARCH="arm64" ;;
        armv7l) ARCH="armv7" ;;
        *) echo "Error: Unsupported architecture: $ARCH"; exit 1 ;;
    esac

    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    GUM_VERSION="v0.17.0"
    GUM_URL="https://github.com/charmbracelet/gum/releases/download/${GUM_VERSION}/gum_${GUM_VERSION#v}_${OS}_${ARCH}.tar.gz"

    TEMP_DIR=$(mktemp -d)
    OLD_DIR=$(pwd)

    if ! curl -sL "$GUM_URL" -o "$TEMP_DIR/gum.tar.gz" 2>/dev/null; then
        echo "Error: Failed to download Gum"
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    cd "$TEMP_DIR"
    tar -xzf gum.tar.gz 2>/dev/null
    GUM_BIN=$(find . -name "gum" -type f | head -n1)

    if [[ -z "$GUM_BIN" ]]; then
        echo "Error: Failed to extract Gum binary"
        cd "$OLD_DIR"
        rm -rf "$TEMP_DIR"
        exit 1
    fi

    chmod +x "$GUM_BIN"
    GUM="$TEMP_DIR/$GUM_BIN"
    cd "$OLD_DIR"

    :
}

setup_gum

if [[ $EUID -ne 0 ]] && [[ "$DRY_RUN" != true ]]; then
    echo ""
    $GUM style --foreground "$C_ERROR" "Error: Root privileges required"
    $GUM style --foreground "$C_MUTED" "Run with: sudo $0"
    exit 1
fi

echo ""
$GUM style \
    --border double --border-foreground "$C_PRIMARY" \
    --padding "1 2" --align center \
    "Stelline SDK Installer" \

echo ""

if [[ "$DRY_RUN" == true ]]; then
    $GUM style --foreground "$C_WARNING" "[DRY RUN MODE - No changes will be made]"
    echo ""
fi

ALREADY_INSTALLED=false
if [[ -f "$CONFIG_FILE" ]]; then
    ALREADY_INSTALLED=true
fi

if [[ "$ALREADY_INSTALLED" == true ]]; then
    echo "Existing installation detected"
    echo ""
    ACTION=$($GUM choose "Reinstall" "Uninstall" "Exit")

    case "$ACTION" in
        "Exit")
            echo "Cancelled"
            exit 0
            ;;
        "Uninstall")
            echo ""
            $GUM style --foreground "$C_WARNING" "This will remove Stelline and its configuration"
            if $GUM confirm "Are you sure?"; then
                systemctl stop stelline 2>/dev/null || true
                systemctl disable stelline 2>/dev/null || true
                rm -f "$SYSTEMD_SERVICE"
                rm -f "$CLI_TOOL"
                rm -rf /etc/stelline
                systemctl daemon-reload 2>/dev/null || true
                echo ""
                $GUM style \
                    --border double --border-foreground "$C_SUCCESS" \
                    --foreground "$C_SUCCESS" \
                    --padding "1 2" --align center \
                    "Stelline uninstalled!"
            else
                echo "Cancelled"
            fi
            exit 0
            ;;
        "Reinstall")
            echo ""
            ;;
    esac
fi

$GUM style --foreground "$C_WHITE" "This tool will install a Docker container with the latest Stelline SDK accessible via Jupyter Notebook."
echo ""
$GUM style --foreground "$C_INFO" "This installer will:"
echo "  - Verify your system meets the requirements"
echo "  - Configure your workspace and settings"
echo "  - Install systemd service"
echo "  - Set up Stelline SDK Host Manager CLI"
echo ""
$GUM style --foreground "$C_INFO" "Documentation:"
echo "  https://stelline.luigi.ltd/quickstart/"
echo ""

if ! $GUM confirm "Begin installation?"; then
    echo "Cancelled"
    exit 0
fi

echo ""
$GUM style \
    --border rounded --border-foreground "$C_PRIMARY" \
    --padding "0 1" \
    "STEP 1: SYSTEM REQUIREMENTS"
echo ""
$GUM style --foreground "$C_MUTED" "Checking if your system meets the requirements for running Stelline."
$GUM style --foreground "$C_MUTED" "This will verify Docker, NVIDIA drivers, and required runtime components."
echo ""

if ! $GUM confirm "Start system check?"; then
    echo "Installation cancelled"
    exit 0
fi
echo ""

check_os() {
    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release 2>/dev/null || true
        local id_lc="${ID,,}"
        local ver="${VERSION_ID:-}"
        if [[ "$id_lc" != "ubuntu" ]]; then
            echo "${NAME:-$ID} (requires Ubuntu >= 22.04)"
            return 1
        fi
        local major minor
        major=${ver%%.*}
        minor=${ver#*.}
        # handle cases like 24.04.1 by trimming after second dot
        minor=${minor%%.*}
        if [[ -z "$major" ]]; then
            echo "Unknown version (requires Ubuntu >= 22.04)"
            return 1
        fi
        if (( major > 22 )) || (( major == 22 && minor >= 4 )); then
            echo "Ubuntu ${VERSION_ID}"
            return 0
        else
            echo "Ubuntu ${VERSION_ID} (requires >= 22.04)"
            return 1
        fi
    else
        echo "Unknown (requires Ubuntu >= 22.04)"
        return 1
    fi
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Not installed"
        return 1
    fi
    docker --version 2>&1 | awk '{print $3}' | tr -d ','
}

check_nvidia_driver() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Not installed"
        return 1
    fi

    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -n1)
    DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
    MIN_MAJOR=$(echo "$MIN_NVIDIA_VERSION" | cut -d. -f1)

    if [[ "$DRIVER_MAJOR" -lt "$MIN_MAJOR" ]]; then
        echo "$DRIVER_VERSION (requires >= $MIN_NVIDIA_VERSION)"
        return 1
    fi

    echo "$DRIVER_VERSION"
}

check_nvidia_docker() {
    if ! sudo docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        echo "Not available"
        return 1
    fi
    echo "Available"
}

check_nvidia_icd() {
    local paths=(
        "/usr/share/vulkan/icd.d/nvidia_icd.json"
        "/etc/vulkan/icd.d/nvidia_icd.json"
        "/usr/local/share/vulkan/icd.d/nvidia_icd.json"
    )

    for path in "${paths[@]}"; do
        if [[ -f "$path" ]]; then
            echo "$path"
            return 0
        fi
    done

    if found=$(find /usr/share /etc -path '*/vulkan/icd.d/nvidia_icd.json' -type f -print -quit 2>/dev/null | head -n1); then
        if [[ -n "$found" ]]; then
            echo "$found"
            return 0
        fi
    fi

    echo "Not found"
    return 1
}

CHECKS_PASSED=true

check_requirement() {
    local name=$1
    local check_func=$2

    if result=$($check_func 2>&1); then
        $GUM style --foreground "$C_SUCCESS" "[OK] $name: $result"
    else
        $GUM style --foreground "$C_ERROR" "[FAIL] $name: $result"
        CHECKS_PASSED=false
    fi
}

check_requirement "Ubuntu (>= 22.04)" check_os
check_requirement "Docker (>= $MIN_DOCKER_VERSION)" check_docker
check_requirement "NVIDIA Driver (>= $MIN_NVIDIA_VERSION)" check_nvidia_driver
check_requirement "NVIDIA Container Toolkit" check_nvidia_docker
check_requirement "NVIDIA ICD JSON" check_nvidia_icd
NVIDIA_ICD=$(check_nvidia_icd 2>/dev/null || echo "")
echo ""

if [[ "$CHECKS_PASSED" == false ]]; then
    $GUM style --foreground "$C_ERROR" "System requirements not met"
    echo ""

    if $GUM confirm "View installation instructions?"; then
        echo ""
        $GUM style --foreground "$C_PRIMARY" --bold "Official Quickstart:"
        echo "  https://stelline.luigi.ltd/quickstart/"
        echo ""
        $GUM style --foreground "$C_PRIMARY" --bold "Additional Resources:"
        echo "  Docker: https://docs.docker.com/engine/install/"
        echo "  NVIDIA Driver: https://www.nvidia.com/download/index.aspx"
        echo "  NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    fi
    exit 1
fi



echo ""
$GUM style \
    --border rounded --border-foreground "$C_PRIMARY" \
    --padding "0 1" \
    "STEP 2: CONFIGURATION"
echo ""
$GUM style --foreground "$C_MUTED" "Configure where Stelline will store your projects and notebooks."
$GUM style --foreground "$C_MUTED" "You can also add additional directories to mount inside the container."
echo ""

DEFAULT_WORKSPACE="$(pwd)/stelline"
$GUM style --foreground "$C_INFO" --bold "Workspace Directory"
WORKSPACE=$($GUM input --placeholder "$DEFAULT_WORKSPACE" --value "$DEFAULT_WORKSPACE" --header "Path:")
$GUM style --foreground "$C_SUCCESS" "  $WORKSPACE"
echo ""

$GUM style --foreground "$C_INFO" --bold "Jupyter Notebook Port"
PORT=$($GUM input --placeholder "8888" --value "8888" --header "Port:")
$GUM style --foreground "$C_SUCCESS" "  $PORT"
echo ""

$GUM style --foreground "$C_INFO" --bold "Auto-start on Boot"
if $GUM confirm "Enable auto-start when system boots?"; then
    ENABLE_AUTOSTART=true
else
    ENABLE_AUTOSTART=false
fi
$GUM style --foreground "$C_SUCCESS" "  $( [[ "$ENABLE_AUTOSTART" == true ]] && echo Enabled || echo Disabled )"
echo ""

$GUM style --foreground "$C_INFO" --bold "Additional Volume Mounts"
EXTRA_VOLUMES=()
if $GUM confirm "Add extra directories to mount?"; then
    echo ""
    $GUM style --foreground "$C_MUTED" "Enter the path to mount. The same path will be used inside the container."
    $GUM style --foreground "$C_MUTED" "Example: /datasets will be mounted as /datasets inside the container"

    while true; do
        echo ""
        HOST_PATH=$($GUM input --placeholder "/path/to/mount (empty to finish)" --header "Directory path:")

        [[ -z "$HOST_PATH" ]] && break

        if [[ ! -d "$HOST_PATH" ]]; then
            $GUM style --foreground "$C_WARNING" "Warning: Directory does not exist: $HOST_PATH"
            if ! $GUM confirm "Add anyway?"; then
                continue
            fi
        fi

        EXTRA_VOLUMES+=("$HOST_PATH:$HOST_PATH")
        echo ""
        $GUM style --foreground "$C_INFO" --bold "Selected Volume Mounts"
        if (( ${#EXTRA_VOLUMES[@]} > 0 )); then
            for vol in "${EXTRA_VOLUMES[@]}"; do
                $GUM style --foreground "$C_SUCCESS" "  - ${vol%%:*}"
            done
        else
            $GUM style --foreground "$C_MUTED" "  None"
        fi
        echo ""

        $GUM confirm "Add another?" || break
    done
fi

echo ""
$GUM style --foreground "$C_INFO" --bold "Selected Volume Mounts"
if (( ${#EXTRA_VOLUMES[@]} > 0 )); then
    for vol in "${EXTRA_VOLUMES[@]}"; do
        $GUM style --foreground "$C_SUCCESS" "  - ${vol%%:*}"
    done
else
    $GUM style --foreground "$C_MUTED" "  None"
fi
echo ""

echo ""
$GUM style \
    --border rounded --border-foreground "$C_PRIMARY" \
    --padding "0 1" \
    "STEP 3: INSTALLATION SUMMARY"
echo ""
$GUM style --foreground "$C_MUTED" "Review your configuration before proceeding with the installation."
echo ""

AUTOSTART_TEXT="Disabled"
[[ "$ENABLE_AUTOSTART" == true ]] && AUTOSTART_TEXT="Enabled"

VOL_ARGS=()
if [[ ${#EXTRA_VOLUMES[@]} -gt 0 ]]; then
    for vol in "${EXTRA_VOLUMES[@]}"; do
        VOL_ARGS+=("$($GUM style --foreground "$C_SUCCESS" "  - ${vol%%:*}")")
    done
else
    VOL_ARGS+=("$($GUM style --foreground "$C_SUCCESS" "  None")")
fi

$GUM style \
    --border double --border-foreground "$C_INFO" \
    --padding "1 2" \
    "$( $GUM style --foreground "$C_PRIMARY" --bold "Installation Configuration" )" \
    "" \
    "$( $GUM style --foreground "$C_INFO" --bold "Workspace Directory:" )" \
    "$( $GUM style --foreground "$C_SUCCESS" "  $WORKSPACE" )" \
    "" \
    "$( $GUM style --foreground "$C_INFO" --bold "Jupyter Notebook Port:" )" \
    "$( $GUM style --foreground "$C_SUCCESS" "  $PORT" )" \
    "" \
    "$( $GUM style --foreground "$C_INFO" --bold "Auto-start on Boot:" )" \
    "$( $GUM style --foreground "$C_SUCCESS" "  $AUTOSTART_TEXT" )" \
    "" \
    "$( $GUM style --foreground "$C_INFO" --bold "Docker Image:" )" \
    "$( $GUM style --foreground "$C_SUCCESS" "  $DOCKER_IMAGE" )" \
    "" \
    "$( $GUM style --foreground "$C_INFO" --bold "Extra Volume Mounts:" )" \
    "${VOL_ARGS[@]}" \
    "" \
    "$( $GUM style --foreground "$C_INFO" --bold "What will be installed:" )" \
    "  - Systemd service for container management" \
    "  - CLI management tool (stelline-host)" \
    "  - Metadata files in /etc/stelline"

echo ""

if [[ "$DRY_RUN" == true ]]; then
    $GUM style --foreground "$C_WARNING" "[DRY RUN - Review mode, no changes will be made]"
    echo ""
fi

if ! $GUM confirm "Proceed with installation?"; then
    echo "Installation cancelled"
    exit 0
fi

echo ""
$GUM style \
    --border rounded --border-foreground "$C_PRIMARY" \
    --padding "0 1" \
    "STEP 4: INSTALLING STELLINE"
echo ""

SCRIPT_PATH="$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null || echo "$0")"

TOTAL_STEPS=3
CURRENT_STEP=0

do_step() {
    local step_name=$1
    local step_func=$2
    local log_file
    log_file=$(mktemp)

    CURRENT_STEP=$((CURRENT_STEP + 1))
    if [[ "$DRY_RUN" == true ]]; then
        $GUM spin --spinner dot --title "Simulating..." -- sleep 0.3
        $GUM style --foreground "$C_WARNING" "[SKIP] [$CURRENT_STEP/$TOTAL_STEPS] $step_name"
    else
        local preamble=""
        if declare -f "$step_func" >/dev/null 2>&1; then
            local var
            for var in WORKSPACE PORT DOCKER_IMAGE NVIDIA_ICD SYSTEMD_SERVICE ENABLE_AUTOSTART CONFIG_FILE CLI_TOOL SCRIPT_PATH; do
                if declare -p "$var" >/dev/null 2>&1; then
                    preamble+=$(declare -p "$var")$'\n'
                fi
            done
            if declare -p EXTRA_VOLUMES >/dev/null 2>&1; then
                preamble+=$(declare -p EXTRA_VOLUMES)$'\n'
            fi
            preamble+=$(declare -f "$step_func")$'\n'
        fi

        local wrapped_cmd="${preamble}set -e; set -o pipefail; { $step_func; } >\"$log_file\" 2>&1"
        if $GUM spin --spinner dot --title "Processing..." -- bash -c "$wrapped_cmd"; then
            $GUM style --foreground "$C_SUCCESS" "[OK] [$CURRENT_STEP/$TOTAL_STEPS] $step_name"
            rm -f "$log_file"
        else
            $GUM style --foreground "$C_ERROR" "[FAIL] [$CURRENT_STEP/$TOTAL_STEPS] $step_name"
            if [[ -s "$log_file" ]]; then
                echo ""
                $GUM style --foreground "$C_ERROR" --bold "Error output:"
                $GUM format < "$log_file"
                echo ""
            fi
            rm -f "$log_file"
            return 1
        fi
    fi
}

do_step "Create workspace directory" "mkdir -p '$WORKSPACE' && sleep 0.3" || exit 1

create_systemd_service() {
    VOLUME_MOUNTS="-v $WORKSPACE:/workspace/stelline"

    if [[ -n "$NVIDIA_ICD" ]]; then
        VOLUME_MOUNTS="$VOLUME_MOUNTS -v $NVIDIA_ICD:$NVIDIA_ICD:ro"
    fi

    for vol in "${EXTRA_VOLUMES[@]}"; do
        VOLUME_MOUNTS="$VOLUME_MOUNTS -v $vol"
    done

    cat > "$SYSTEMD_SERVICE" <<EOF
[Unit]
Description=Stelline Development Environment
After=docker.service
Requires=docker.service

[Service]
Type=simple
Restart=always
RestartSec=10
ExecStart=/usr/bin/docker run --rm --name stelline \\
    --net host \\
    --privileged \\
    --gpus=all \\
    --cap-add CAP_SYS_PTRACE \\
    --ipc=host \\
    --volume /run/udev:/run/udev:ro \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    -v /dev/bus/usb:/dev/bus/usb \\
    -v /tmp/.X11-unix:/tmp/.X11-unix \\
    $VOLUME_MOUNTS \\
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \\
    -e DISPLAY=\$DISPLAY \\
    -p $PORT:8888 \\
    $DOCKER_IMAGE
ExecStop=/usr/bin/docker stop stelline

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload

    if [[ "$ENABLE_AUTOSTART" == true ]]; then
        systemctl enable --now stelline
    fi

    return 0
}

do_step "Create systemd service" "create_systemd_service" || exit 1

install_cli_tool() {
    cat > "$CLI_TOOL" <<'EOFCLI'
#!/bin/bash

WORKSPACE="WORKSPACE_PLACEHOLDER"
PORT="PORT_PLACEHOLDER"

case "$1" in
    start)
        echo "Starting Stelline..."
        sudo systemctl start stelline
        echo "[OK] Started"
        echo ""
        echo "Access Jupyter Notebook:"
        echo "  http://localhost:$PORT"
        ;;

    stop)
        echo "Stopping Stelline..."
        sudo systemctl stop stelline
        echo "[OK] Stopped"
        ;;

    restart)
        echo "Restarting Stelline..."
        sudo systemctl restart stelline
        echo "[OK] Restarted"
        echo ""
        echo "Access Jupyter Notebook:"
        echo "  http://localhost:$PORT"
        ;;

    status)
        systemctl status stelline
        ;;

    logs)
        if [[ "$2" == "-f" ]]; then
            journalctl -u stelline -f
        else
            journalctl -u stelline -n 50
        fi
        ;;

    installer)
        echo "Launching installer..."
        if [[ -x /etc/stelline/installer.sh ]]; then
            sudo /etc/stelline/installer.sh
        else
            echo "Installer not found at /etc/stelline/installer.sh"
            echo "Re-download the installer or reinstall to recreate it."
            exit 1
        fi
        ;;

    help|--help|-h|"")
        echo "Stelline SDK Host Manager"
        echo ""
        echo "Usage: stelline-host [command]"
        echo ""
        echo "Commands:"
        echo "  start      - Start service"
        echo "  stop       - Stop service"
        echo "  restart    - Restart service"
        echo "  status     - Show service status"
        echo "  logs       - Show logs (use -f to follow)"
        echo "  installer  - Run installer or uninstaller"
        echo "  help       - Show this help"
        echo ""
        echo "Access:"
        echo "  Workspace: $WORKSPACE"
        echo "  Notebook:  http://localhost:$PORT"
        echo ""
        echo "Documentation:"
        echo "  https://stelline.luigi.ltd/quickstart/"
        echo ""
        ;;

    *)
        echo "Unknown command: $1"
        echo "Run 'stelline-host help' for usage"
        exit 1
        ;;
esac
EOFCLI

    sed -i "s|WORKSPACE_PLACEHOLDER|$WORKSPACE|g" "$CLI_TOOL"
    sed -i "s|PORT_PLACEHOLDER|$PORT|g" "$CLI_TOOL"

    chmod +x "$CLI_TOOL"

    mkdir -p "$(dirname -- "$CONFIG_FILE")"
    mkdir -p /etc/stelline 2>/dev/null || true
    cp -f "$SCRIPT_PATH" /etc/stelline/installer.sh 2>/dev/null || true
    chmod +x /etc/stelline/installer.sh 2>/dev/null || true
    cat > "$CONFIG_FILE" <<EOF
{
  "workspace_path": "$WORKSPACE",
  "docker_image": "$DOCKER_IMAGE",
  "port": $PORT,
  "nvidia_icd_json": "$NVIDIA_ICD",
  "extra_volumes": [
$(printf '    "%s",\n' "${EXTRA_VOLUMES[@]}" | sed '$ s/,$//')
  ],
  "autostart_enabled": $ENABLE_AUTOSTART
}
EOF

    return 0
}

do_step "Install CLI tool" "install_cli_tool" || exit 1

echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    $GUM style --foreground "$C_WARNING" "Dry-run complete - No changes were made"
    echo ""
    echo "To perform actual installation:"
    echo "  sudo $0"
else
    echo ""
    IP_URLS=()
    if command -v ip >/dev/null 2>&1; then
        while IFS= read -r ipaddr; do
            IP_URLS+=("  http://$ipaddr:$PORT")
        done < <(ip -o -4 addr show up | awk '{print $4}' | cut -d/ -f1)
    fi
    $GUM style \
        --border double --border-foreground "$C_SUCCESS" \
        --padding "1 2" --align left \
        "$( $GUM style --foreground "$C_PRIMARY" --bold "Stelline SDK installation complete!" )" \
        "" \
        "A Docker container with the latest Stelline SDK was created." \
        "" \
        "$( $GUM style --foreground "$C_INFO" --bold "Navigate to the Jupyter Notebook to get started:" )" \
        "${IP_URLS[@]}" \
        "" \
        "$( $GUM style --foreground "$C_INFO" --bold "Additional Documentation:" )" \
        "  https://stelline.luigi.ltd/quickstart/" \
        "" \
        "$( $GUM style --foreground "$C_INFO" --bold "View all management commands:" )" \
        "  stelline-host help"
fi
