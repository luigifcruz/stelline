import { defineConfig } from "vitepress";

export default defineConfig({
    title: "Stelline",
    titleTemplate: ":title - Stelline",
    description: "Software Defined Observatory SDK",
    cleanUrls: true,
    themeConfig: {
        outline: [2, 3],
        nav: [
            {
                text: "Overview",
                link: "/overview/",
                activeMatch: "^/overview/",
            },
            {
                text: "Quickstart",
                link: "/quickstart/",
                activeMatch: "^/quickstart/",
            },
            {
                text: "SDK Guide",
                link: "/sdk-guide/",
                activeMatch: "^/sdk-guide/",
            },
        ],
        sidebar: {
            "/overview/": [
                {
                    text: "Overview",
                    link: "/overview/",
                    items: [
                        {
                            text: "Design Concepts",
                            link: "/overview/design-concepts",
                        },
                        { text: "Deployments", link: "/overview/deployments" },
                        { text: "Papers & Talks", link: "/overview/talks" },
                        { text: "Frequent Questions", link: "/overview/faq" },
                        {
                            text: "Acknowledgements",
                            link: "/overview/acknowledgements",
                        },
                        { text: "License", link: "/overview/license" },
                    ],
                },
            ],
            "/quickstart/": [
                {
                    text: "Quickstart",
                    link: "/quickstart/",
                    items: [
                        {
                            text: "Requirements",
                            link: "/quickstart/requirements",
                        },
                        {
                            text: "Considerations",
                            link: "/quickstart/considerations",
                        },
                        {
                            text: "Configuration",
                            link: "/quickstart/configuration",
                        },
                        {
                            text: "Dependencies",
                            link: "/quickstart/dependencies",
                        },
                        {
                            text: "Installation",
                            link: "/quickstart/installation",
                        },
                    ],
                },
                {
                    text: "Examples",
                    items: [
                        {
                            text: "Hello World",
                            link: "/quickstart/examples/hello-world",
                        },
                        {
                            text: "Minimal Streaming",
                            link: "/quickstart/examples/minimal-streaming",
                        },
                        {
                            text: "Allen Telescope Array",
                            collapsed: true,
                            items: [
                                {
                                    text: "Correlator",
                                    link: "/quickstart/examples/ata/correlator",
                                },
                                {
                                    text: "Beamformer",
                                    link: "/quickstart/examples/ata/beamformer",
                                },
                                {
                                    text: "FRBNN",
                                    link: "/quickstart/examples/ata/frbnn",
                                },
                            ],
                        },
                    ],
                },
                {
                    text: "User Guide",
                    items: [
                        {
                            text: "Command Line (CLI)",
                            link: "/quickstart/user-guide/command-line",
                        },
                        {
                            text: "Recipes (YAML)",
                            link: "/quickstart/user-guide/recipes",
                        },
                    ],
                },
                {
                    text: "Validated Hardware",
                    link: "/quickstart/validated-hardware/",
                    items: [
                        {
                            text: "Servers",
                            collapsed: true,
                            items: [
                                {
                                    text: "NVIDIA IGX Orin",
                                    link: "/quickstart/validated-hardware/server/nvidia-igx-orin",
                                },
                                {
                                    text: "Supermicro 4125GS-TNRT1",
                                    link: "/quickstart/validated-hardware/server/supermicro-4125gs-tnrt1",
                                },
                                {
                                    text: "Supermicro 4125GS-TNRT",
                                    link: "/quickstart/validated-hardware/server/supermicro-4125gs-tnrt",
                                },
                                {
                                    text: "ASUS ESC8000A-E12",
                                    link: "/quickstart/validated-hardware/server/asus-esc8000a-e12",
                                },
                                {
                                    text: "Puget TR PRO WRX90",
                                    link: "/quickstart/validated-hardware/server/puget-threadripper-pro-wrx90",
                                },
                            ],
                        },
                        {
                            text: "NVIDIA GPUs",
                            collapsed: true,
                            items: [
                                {
                                    text: "NVIDIA RTX A6000",
                                    link: "/quickstart/validated-hardware/gpu/nvidia-rtx-a6000",
                                },
                                {
                                    text: "NVIDIA RTX 6000 Ada",
                                    link: "/quickstart/validated-hardware/gpu/nvidia-rtx-6000-ada",
                                },
                            ],
                        },
                        {
                            text: "NVMe",
                            collapsed: true,
                            items: [
                                {
                                    text: "Crucial T700",
                                    link: "/quickstart/validated-hardware/nvme/crucial-t700",
                                },
                                {
                                    text: "Samsung 990 Pro",
                                    link: "/quickstart/validated-hardware/nvme/samsung-990-pro",
                                },
                                {
                                    text: "Samsung 9100 Pro",
                                    link: "/quickstart/validated-hardware/nvme/samsung-9100-pro",
                                },
                            ],
                        },
                        {
                            text: "NVMe Carrier Boards",
                            collapsed: true,
                            items: [
                                {
                                    text: "HighPoint SSD7540",
                                    link: "/quickstart/validated-hardware/nvme-carrier-board/highpoint-ssd7540",
                                },
                                {
                                    text: "HighPoint Rocket 7608A",
                                    link: "/quickstart/validated-hardware/nvme-carrier-board/highpoint-rocket-7608a",
                                },
                            ],
                        },
                    ],
                },
            ],
            "/sdk-guide/": [
                {
                    text: "SDK Guide",
                    link: "/sdk-guide/",
                    items: [
                        {
                            text: "Contributing Guide",
                            link: "/sdk-guide/contributing-guide",
                        },
                        {
                            text: "Metadata Sharing",
                            link: "/sdk-guide/metadata-sharing",
                        },
                        {
                            text: "Release Notes",
                            link: "/sdk-guide/release-notes",
                        },
                    ],
                },
                {
                    text: "Bits Reference",
                    collapsed: false,
                    items: [
                        {
                            text: "BLADE",
                            link: "/sdk-guide/bits-reference/blade",
                        },
                        {
                            text: "Filesystem",
                            link: "/sdk-guide/bits-reference/io",
                        },
                        {
                            text: "Transport",
                            link: "/sdk-guide/bits-reference/transport",
                        },
                        {
                            text: "FRBNN",
                            link: "/sdk-guide/bits-reference/frbnn",
                        },
                        {
                            text: "Socket",
                            link: "/sdk-guide/bits-reference/socket",
                        },
                    ],
                },
                {
                    text: "Operator Reference",
                    collapsed: false,
                    items: [
                        {
                            text: "BLADE",
                            link: "/sdk-guide/operator-reference/blade",
                        },
                        {
                            text: "Filesystem",
                            link: "/sdk-guide/operator-reference/io",
                        },
                        {
                            text: "Transport",
                            link: "/sdk-guide/operator-reference/transport",
                        },
                        {
                            text: "FRBNN",
                            link: "/sdk-guide/operator-reference/frbnn",
                        },
                        {
                            text: "Socket",
                            link: "/sdk-guide/operator-reference/socket",
                        },
                    ],
                },
                { text: "Troubleshooting", link: "/sdk-guide/troubleshooting" },
            ],
        },
        socialLinks: [
            { icon: "github", link: "https://github.com/luigifcruz/stelline" },
        ],
    },
});
