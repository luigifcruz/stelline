#ifndef STELLINE_NEXUS_BRIDGE_BLOCK_HH
#define STELLINE_NEXUS_BRIDGE_BLOCK_HH

#include <string>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct NexusBridge : public Block::Config {
    std::string url = "https://nexus.stelline.space";

    JST_BLOCK_TYPE(stelline_nexus_bridge);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(url);
    JST_BLOCK_DESCRIPTION(
        "Nexus Bridge",
        "Syncs Nexus metadata and publishes block metrics.",
        "# Nexus Bridge\n"
        "\n"
        "Mirrors the metadata that Nexus holds for this instance into the flowgraph "
        "environment, and reports Stelline block metrics back to Nexus. It sits alone "
        "on the canvas, next to writers such as the UVH5 Writer that read the "
        "observatory and observation keys.\n"
        "\n"
        "- **Credentials come from the process environment.** Set `NEXUS_INSTANCE_ID` "
        "and a credential variable before starting CyberEther, as in Setup.\n"
        "- **Entries keep their Nexus form.** Each key holds a dict with `value`, "
        "`type`, and `valid`, and integers arrive as decimal strings.\n"
        "- **Both directions run every 5 seconds.** Metadata is polled and the hidden "
        "metrics of Stelline blocks are uploaded on that interval.\n"
        "- **Values stay while disconnected.** A failed request keeps the last "
        "metadata, and only keys that Nexus removes are dropped.\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Nexus URL** | https://nexus.stelline.space | Not empty | Nexus deployment "
        "to reach. Only its scheme and host are used, with `/telemetry` added. Ignored "
        "when `NEXUS_TELEMETRY_URL` is set. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **Connected** | Connected when the latest metadata request succeeded, "
        "Disconnected before the first success and after a failure. |\n"
        "| **Variables Loaded** | Metadata keys mirrored into the environment. |\n"
        "| **Metrics Monitored** | Hidden Stelline block metrics in the latest "
        "snapshot. Stays 0 without `NEXUS_INSTANCE_ID`. |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| Connected stays Disconnected, and `last_error` under `nexus.bridge` in the "
        "Environment window says `NEXUS_INSTANCE_ID` or the credentials are required | "
        "The variable is not set in the process that started CyberEther | Set it as in "
        "Setup and restart CyberEther. |\n"
        "| The `last_error` field names a URLError or a TimeoutError | The host of "
        "Nexus URL, or `NEXUS_TELEMETRY_URL`, cannot be reached | Check the address and"
        " the network path to it. |\n"
        "| The `last_error` field shows an HTTP error | Nexus rejects the instance or "
        "its token, such as after the observation ended | Renew the token in the "
        "credentials file, which is read before every request. |\n"
        "| A writer reports a missing `observatory.name` environment value | The bridge"
        " has not connected yet, or Nexus has no metadata for this instance | Wait for "
        "Connected and a Variables Loaded above 0. |\n"
        "| The block reports that the Nexus URL must not be empty | Nexus URL was "
        "cleared | Enter the deployment URL. |\n"
        "\n"
        "## Setup\n"
        "\n"
        "| Platform | Requirement |\n"
        "|---|---|\n"
        "| Python runtime | Python 3.9 or newer selected under Settings, Runtime, "
        "Python Runtime. Only the standard library is used. |\n"
        "| Instance | The ID in `NEXUS_INSTANCE_ID` names the instance whose metadata "
        "is read and whose metrics are written. Required. |\n"
        "| Credentials | A bearer token in the file named by "
        "`NEXUS_INSTANCE_CREDENTIALS_FILE`, read first, or in "
        "`NEXUS_INSTANCE_CREDENTIALS`. A file that cannot be read stops the requests. |\n"
        "| Telemetry address | Optional. An address in `NEXUS_TELEMETRY_URL` replaces "
        "the one built from Nexus URL. |\n"
        "| Retry delay | Optional. Seconds between metadata retries, read from "
        "`NEXUS_OTL_WATCH_RETRY_SECONDS`, 5 by default. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "A background thread polls Nexus and queues the metadata that changed, so "
        "compute never waits on the network. Each throttled cycle applies the queued "
        "changes and publishes the bridge status under `nexus.bridge`. A second thread "
        "uploads metric snapshots in batches, retries them on network or server errors,"
        " and drops them on other rejections.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- The [Nexus Bridge guide](https://stelline.space/docs/block-nexus-bridge) "
        "covers the Nexus service and its credentials.\n"
        "- Observatory and observation keys feed the **UVH5 Writer**.\n"
        "- Observatory and pointing keys feed the **FBH5 Writer**.\n"
        "- Uploaded metrics come from blocks such as the **ATA Receiver**."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_NEXUS_BRIDGE_BLOCK_HH
