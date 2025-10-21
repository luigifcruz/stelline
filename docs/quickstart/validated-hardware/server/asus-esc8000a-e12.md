# ASUS ESC8000A-E12

## PCIe Topology
```
$ sonata@coyote2:~$ lspci -tv
-+-[0000:00]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14a4
 |           +-00.3  Advanced Micro Devices, Inc. [AMD] Device 14a6
 |           +-01.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-01.1-[01]--+-00.0  NVIDIA Corporation AD102GL [RTX 6000 Ada Generation]
 |           |            \-00.1  NVIDIA Corporation AD102 High Definition Audio Controller
 |           +-02.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-03.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-04.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-05.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-05.1-[02]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller SM981/PM981/PM983
 |           +-07.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-07.1-[03]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14ac
 |           |            +-00.1  Advanced Micro Devices, Inc. [AMD] Device 14dc
 |           |            +-00.4  Advanced Micro Devices, Inc. [AMD] Device 14c9
 |           |            \-00.5  Advanced Micro Devices, Inc. [AMD] Genoa CCP/PSP 4.0 Device
 |           +-14.0  Advanced Micro Devices, Inc. [AMD] FCH SMBus Controller
 |           +-14.3  Advanced Micro Devices, Inc. [AMD] FCH LPC Bridge
 |           +-18.0  Advanced Micro Devices, Inc. [AMD] Device 14ad
 |           +-18.1  Advanced Micro Devices, Inc. [AMD] Device 14ae
 |           +-18.2  Advanced Micro Devices, Inc. [AMD] Device 14af
 |           +-18.3  Advanced Micro Devices, Inc. [AMD] Device 14b0
 |           +-18.4  Advanced Micro Devices, Inc. [AMD] Device 14b1
 |           +-18.5  Advanced Micro Devices, Inc. [AMD] Device 14b2
 |           +-18.6  Advanced Micro Devices, Inc. [AMD] Device 14b3
 |           +-18.7  Advanced Micro Devices, Inc. [AMD] Device 14b4
 |           +-19.0  Advanced Micro Devices, Inc. [AMD] Device 14ad
 |           +-19.1  Advanced Micro Devices, Inc. [AMD] Device 14ae
 |           +-19.2  Advanced Micro Devices, Inc. [AMD] Device 14af
 |           +-19.3  Advanced Micro Devices, Inc. [AMD] Device 14b0
 |           +-19.4  Advanced Micro Devices, Inc. [AMD] Device 14b1
 |           +-19.5  Advanced Micro Devices, Inc. [AMD] Device 14b2
 |           +-19.6  Advanced Micro Devices, Inc. [AMD] Device 14b3
 |           \-19.7  Advanced Micro Devices, Inc. [AMD] Device 14b4
 +-[0000:20]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14a4
 |           +-00.3  Advanced Micro Devices, Inc. [AMD] Device 14a6
 |           +-01.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-02.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-03.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-03.1-[21]--+-00.0  Mellanox Technologies MT2910 Family [ConnectX-7]
 |           |            \-00.1  Mellanox Technologies MT2910 Family [ConnectX-7]
 |           +-04.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-05.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-07.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           \-07.1-[22]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14ac
 |                        \-00.1  Advanced Micro Devices, Inc. [AMD] Device 14dc
 +-[0000:40]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14a4
 |           +-00.3  Advanced Micro Devices, Inc. [AMD] Device 14a6
 |           +-01.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-01.1-[41]--+-00.0  NVIDIA Corporation AD102GL [RTX 6000 Ada Generation]
 |           |            \-00.1  NVIDIA Corporation AD102 High Definition Audio Controller
 |           +-02.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-03.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-04.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-05.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-07.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           \-07.1-[42]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14ac
 |                        \-00.1  Advanced Micro Devices, Inc. [AMD] Device 14dc
 +-[0000:60]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14a4
 |           +-00.3  Advanced Micro Devices, Inc. [AMD] Device 14a6
 |           +-01.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-01.1-[61-73]----00.0-[62-73]--+-00.0-[63-68]----00.0-[64-68]--+-10.0-[65]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               +-14.0-[66]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               +-18.0-[67]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               \-1c.0-[68]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               +-04.0-[69-6e]----00.0-[6a-6e]--+-00.0-[6b]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               +-04.0-[6c]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               +-08.0-[6d]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               \-0c.0-[6e]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               +-0c.0-[6f-72]----00.0-[70-72]--+-14.0-[71]----00.0  HighPoint Technologies, Inc. SSD7540 PCIe Gen4 x16 8-Port M.2 NVMe RAID Controller
 |           |                               |                               \-15.0-[72]--
 |           |                               \-1c.0-[73]--
 |           +-02.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-03.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-04.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-05.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-05.2-[74-75]----00.0-[75]----00.0  ASPEED Technology, Inc. ASPEED Graphics Family
 |           +-07.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           \-07.1-[76]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14ac
 |                        +-00.1  Advanced Micro Devices, Inc. [AMD] Device 14dc
 |                        \-00.4  Advanced Micro Devices, Inc. [AMD] Device 14c9
 +-[0000:80]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14a4
 |           +-00.3  Advanced Micro Devices, Inc. [AMD] Device 14a6
 |           +-01.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-01.1-[81]--+-00.0  Mellanox Technologies MT2910 Family [ConnectX-7]
 |           |            \-00.1  Mellanox Technologies MT2910 Family [ConnectX-7]
 |           +-02.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-03.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-04.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-05.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-05.1-[82-83]--+-00.0  Intel Corporation Ethernet Controller X710 for 10GBASE-T
 |           |               \-00.1  Intel Corporation Ethernet Controller X710 for 10GBASE-T
 |           +-07.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           \-07.1-[84]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14ac
 |                        +-00.1  Advanced Micro Devices, Inc. [AMD] Device 14dc
 |                        +-00.4  Advanced Micro Devices, Inc. [AMD] Device 14c9
 |                        \-00.5  Advanced Micro Devices, Inc. [AMD] Genoa CCP/PSP 4.0 Device
 +-[0000:a0]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14a4
 |           +-00.3  Advanced Micro Devices, Inc. [AMD] Device 14a6
 |           +-01.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-01.1-[a1-b3]----00.0-[a2-b3]--+-00.0-[a3-a8]----00.0-[a4-a8]--+-10.0-[a5]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               +-14.0-[a6]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               +-18.0-[a7]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               \-1c.0-[a8]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               +-04.0-[a9-ae]----00.0-[aa-ae]--+-00.0-[ab]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               +-04.0-[ac]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               +-08.0-[ad]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               \-0c.0-[ae]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               +-0c.0-[af-b2]----00.0-[b0-b2]--+-14.0-[b1]----00.0  HighPoint Technologies, Inc. SSD7540 PCIe Gen4 x16 8-Port M.2 NVMe RAID Controller
 |           |                               |                               \-15.0-[b2]--
 |           |                               \-1c.0-[b3]--
 |           +-02.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-03.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-04.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-05.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-07.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           \-07.1-[b4]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14ac
 |                        \-00.1  Advanced Micro Devices, Inc. [AMD] Device 14dc
 +-[0000:c0]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14a4
 |           +-00.3  Advanced Micro Devices, Inc. [AMD] Device 14a6
 |           +-01.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-01.1-[c1]--+-00.0  NVIDIA Corporation AD102GL [RTX 6000 Ada Generation]
 |           |            \-00.1  NVIDIA Corporation AD102 High Definition Audio Controller
 |           +-02.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-03.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-03.2-[c2]--
 |           +-03.3-[c3]--
 |           +-04.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-05.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-07.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           \-07.1-[c4]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14ac
 |                        \-00.1  Advanced Micro Devices, Inc. [AMD] Device 14dc
 \-[0000:e0]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14a4
             +-00.3  Advanced Micro Devices, Inc. [AMD] Device 14a6
             +-01.0  Advanced Micro Devices, Inc. [AMD] Device 149f
             +-01.1-[e1]--+-00.0  NVIDIA Corporation AD102GL [RTX 6000 Ada Generation]
             |            \-00.1  NVIDIA Corporation AD102 High Definition Audio Controller
             +-02.0  Advanced Micro Devices, Inc. [AMD] Device 149f
             +-03.0  Advanced Micro Devices, Inc. [AMD] Device 149f
             +-04.0  Advanced Micro Devices, Inc. [AMD] Device 149f
             +-05.0  Advanced Micro Devices, Inc. [AMD] Device 149f
             +-05.2-[e2]----00.0  HighPoint Technologies, Inc. SSD7540 PCIe Gen4 x16 8-Port M.2 NVMe RAID Controller
             +-07.0  Advanced Micro Devices, Inc. [AMD] Device 149f
             \-07.1-[e3]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14ac
                          +-00.1  Advanced Micro Devices, Inc. [AMD] Device 14dc
                          \-00.4  Advanced Micro Devices, Inc. [AMD] Device 14c9
```

## NUMA Topology
```
$ sonata@coyote2:~$ numactl --hardware
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
node 0 size: 193129 MB
node 0 free: 186031 MB
node 1 cpus: 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
node 1 size: 193477 MB
node 1 free: 186364 MB
node distances:
node   0   1
  0:  10  32
  1:  32  10
```

## PCIe Devices Details
GPU #0
```
$ sonata@coyote2:~$ sudo lspci -vvv -s 01:00.0
01:00.0 VGA compatible controller: NVIDIA Corporation AD102GL [RTX 6000 Ada Generation] (rev a1) (prog-if 00 [VGA controller])
        Subsystem: NVIDIA Corporation AD102GL [RTX 6000 Ada Generation]
        Physical Slot: 3
        Control: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0
        Interrupt: pin A routed to IRQ 852
        NUMA node: 0
        Region 0: Memory at f8000000 (32-bit, non-prefetchable) [size=16M]
        Region 1: Memory at 3f000000000 (64-bit, prefetchable) [size=64G]
        Region 3: Memory at 40000000000 (64-bit, prefetchable) [size=32M]
        Region 5: I/O ports at 2000 [size=128]
        Expansion ROM at f9000000 [virtual] [disabled] [size=512K]
        Capabilities: [60] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=0mA PME(D0+,D1-,D2-,D3hot+,D3cold-)
                Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [68] MSI: Enable+ Count=1/1 Maskable- 64bit+
                Address: 00000000fee0e000  Data: 0037
        Capabilities: [78] Express (v2) Legacy Endpoint, MSI 00
                DevCap: MaxPayload 256 bytes, PhantFunc 0, Latency L0s unlimited, L1 <64us
                        ExtTag+ AttnBtn- AttnInd- PwrInd- RBE+ FLReset+
                DevCtl: CorrErr+ NonFatalErr+ FatalErr+ UnsupReq-
                        RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop+ FLReset-
                        MaxPayload 256 bytes, MaxReadReq 512 bytes
                DevSta: CorrErr- NonFatalErr- FatalErr- UnsupReq- AuxPwr- TransPend-
                LnkCap: Port #0, Speed 16GT/s, Width x16, ASPM L1, Exit Latency L1 <4us
                        ClockPM+ Surprise- LLActRep- BwNot- ASPMOptComp+
                LnkCtl: ASPM Disabled; RCB 64 bytes, Disabled- CommClk+
                        ExtSynch- ClockPM+ AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 2.5GT/s (downgraded), Width x16
                        TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-
                DevCap2: Completion Timeout: Range AB, TimeoutDis+ NROPrPrP- LTR+
                         10BitTagComp+ 10BitTagReq+ OBFF Via message, ExtFmt- EETLPPrefix-
                         EmergencyPowerReduction Not Supported, EmergencyPowerReductionInit-
                         FRS-
                         AtomicOpsCap: 32bit- 64bit- 128bitCAS-
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis- LTR+ 10BitTagReq- OBFF Via WAKE#,
                         AtomicOpsCtl: ReqEn-
                LnkCap2: Supported Link Speeds: 2.5-16GT/s, Crosslink- Retimer+ 2Retimers+ DRS-
                LnkCtl2: Target Link Speed: 16GT/s, EnterCompliance- SpeedDis-
                         Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
                         Compliance Preset/De-emphasis: -6dB de-emphasis, 0dB preshoot
                LnkSta2: Current De-emphasis Level: -6dB, EqualizationComplete+ EqualizationPhase1+
                         EqualizationPhase2+ EqualizationPhase3+ LinkEqualizationRequest-
                         Retimer- 2Retimers- CrosslinkRes: unsupported
        Capabilities: [b4] Vendor Specific Information: Len=14 <?>
        Capabilities: [100 v1] Virtual Channel
                Caps:   LPEVC=0 RefClk=100ns PATEntryBits=1
                Arb:    Fixed- WRR32- WRR64- WRR128-
                Ctrl:   ArbSelect=Fixed
                Status: InProgress-
                VC0:    Caps:   PATOffset=00 MaxTimeSlots=1 RejSnoopTrans-
                        Arb:    Fixed- WRR32- WRR64- WRR128- TWRR128- WRR256-
                        Ctrl:   Enable+ ID=0 ArbSelect=Fixed TC/VC=01
                        Status: NegoPending- InProgress-
        Capabilities: [250 v1] Latency Tolerance Reporting
                Max snoop latency: 34326183936ns
                Max no snoop latency: 34326183936ns
        Capabilities: [258 v1] L1 PM Substates
                L1SubCap: PCI-PM_L1.2+ PCI-PM_L1.1+ ASPM_L1.2- ASPM_L1.1+ L1_PM_Substates+
                          PortCommonModeRestoreTime=255us PortTPowerOnTime=10us
                L1SubCtl1: PCI-PM_L1.2- PCI-PM_L1.1- ASPM_L1.2- ASPM_L1.1-
                           T_CommonMode=0us
                L1SubCtl2: T_PwrOn=10us
        Capabilities: [128 v1] Power Budgeting <?>
        Capabilities: [420 v2] Advanced Error Reporting
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq+ ACSViol-
                UESvrt: DLP+ SDES+ TLP- FCP+ CmpltTO+ CmpltAbrt- UnxCmplt+ RxOF+ MalfTLP+ ECRC- UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                CEMsk:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                AERCap: First Error Pointer: 00, ECRCGenCap- ECRCGenEn- ECRCChkCap- ECRCChkEn-
                        MultHdrRecCap- MultHdrRecEn- TLPPfxPres- HdrLogCap-
                HeaderLog: 00000000 00000000 00000000 00000000
        Capabilities: [600 v1] Vendor Specific Information: ID=0001 Rev=1 Len=024 <?>
        Capabilities: [900 v1] Secondary PCI Express
                LnkCtl3: LnkEquIntrruptEn- PerformEqu-
                LaneErrStat: 0
        Capabilities: [bb0 v1] Physical Resizable BAR
                BAR 0: current size: 16MB, supported: 16MB
                BAR 1: current size: 64GB, supported: 64MB 128MB 256MB 512MB 1GB 2GB 4GB 8GB 16GB 32GB 64GB
                BAR 3: current size: 32MB, supported: 32MB
        Capabilities: [c1c v1] Physical Layer 16.0 GT/s <?>
        Capabilities: [d00 v1] Lane Margining at the Receiver <?>
        Capabilities: [e00 v1] Data Link Feature <?>
        Kernel driver in use: nvidia
        Kernel modules: nvidiafb, nouveau, nvidia_drm, nvidia
```

GPU #1
```
$ sonata@coyote2:~$ sudo lspci -vvv -s 41:00.0
41:00.0 VGA compatible controller: NVIDIA Corporation AD102GL [RTX 6000 Ada Generation] (rev a1) (prog-if 00 [VGA controller])
        Subsystem: NVIDIA Corporation AD102GL [RTX 6000 Ada Generation]
        Physical Slot: 2
        Control: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0
        Interrupt: pin A routed to IRQ 851
        NUMA node: 0
        Region 0: Memory at f2000000 (32-bit, non-prefetchable) [size=16M]
        Region 1: Memory at 2f000000000 (64-bit, prefetchable) [size=64G]
        Region 3: Memory at 30000000000 (64-bit, prefetchable) [size=32M]
        Region 5: I/O ports at 5000 [size=128]
        Expansion ROM at f3000000 [virtual] [disabled] [size=512K]
        Capabilities: [60] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=0mA PME(D0+,D1-,D2-,D3hot+,D3cold-)
                Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [68] MSI: Enable+ Count=1/1 Maskable- 64bit+
                Address: 00000000fee0d000  Data: 0037
        Capabilities: [78] Express (v2) Legacy Endpoint, MSI 00
                DevCap: MaxPayload 256 bytes, PhantFunc 0, Latency L0s unlimited, L1 <64us
                        ExtTag+ AttnBtn- AttnInd- PwrInd- RBE+ FLReset+
                DevCtl: CorrErr+ NonFatalErr+ FatalErr+ UnsupReq-
                        RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop+ FLReset-
                        MaxPayload 256 bytes, MaxReadReq 512 bytes
                DevSta: CorrErr- NonFatalErr- FatalErr- UnsupReq- AuxPwr- TransPend-
                LnkCap: Port #0, Speed 16GT/s, Width x16, ASPM L1, Exit Latency L1 <4us
                        ClockPM+ Surprise- LLActRep- BwNot- ASPMOptComp+
                LnkCtl: ASPM Disabled; RCB 64 bytes, Disabled- CommClk+
                        ExtSynch- ClockPM+ AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 2.5GT/s (downgraded), Width x16
                        TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-
                DevCap2: Completion Timeout: Range AB, TimeoutDis+ NROPrPrP- LTR+
                         10BitTagComp+ 10BitTagReq+ OBFF Via message, ExtFmt- EETLPPrefix-
                         EmergencyPowerReduction Not Supported, EmergencyPowerReductionInit-
                         FRS-
                         AtomicOpsCap: 32bit- 64bit- 128bitCAS-
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis- LTR+ 10BitTagReq- OBFF Via WAKE#,
                         AtomicOpsCtl: ReqEn-
                LnkCap2: Supported Link Speeds: 2.5-16GT/s, Crosslink- Retimer+ 2Retimers+ DRS-
                LnkCtl2: Target Link Speed: 16GT/s, EnterCompliance- SpeedDis-
                         Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
                         Compliance Preset/De-emphasis: -6dB de-emphasis, 0dB preshoot
                LnkSta2: Current De-emphasis Level: -6dB, EqualizationComplete+ EqualizationPhase1+
                         EqualizationPhase2+ EqualizationPhase3+ LinkEqualizationRequest-
                         Retimer- 2Retimers- CrosslinkRes: unsupported
        Capabilities: [b4] Vendor Specific Information: Len=14 <?>
        Capabilities: [100 v1] Virtual Channel
                Caps:   LPEVC=0 RefClk=100ns PATEntryBits=1
                Arb:    Fixed- WRR32- WRR64- WRR128-
                Ctrl:   ArbSelect=Fixed
                Status: InProgress-
                VC0:    Caps:   PATOffset=00 MaxTimeSlots=1 RejSnoopTrans-
                        Arb:    Fixed- WRR32- WRR64- WRR128- TWRR128- WRR256-
                        Ctrl:   Enable+ ID=0 ArbSelect=Fixed TC/VC=01
                        Status: NegoPending- InProgress-
        Capabilities: [250 v1] Latency Tolerance Reporting
                Max snoop latency: 34326183936ns
                Max no snoop latency: 34326183936ns
        Capabilities: [258 v1] L1 PM Substates
                L1SubCap: PCI-PM_L1.2+ PCI-PM_L1.1+ ASPM_L1.2- ASPM_L1.1+ L1_PM_Substates+
                          PortCommonModeRestoreTime=255us PortTPowerOnTime=10us
                L1SubCtl1: PCI-PM_L1.2- PCI-PM_L1.1- ASPM_L1.2- ASPM_L1.1-
                           T_CommonMode=0us
                L1SubCtl2: T_PwrOn=10us
        Capabilities: [128 v1] Power Budgeting <?>
        Capabilities: [420 v2] Advanced Error Reporting
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq+ ACSViol-
                UESvrt: DLP+ SDES+ TLP- FCP+ CmpltTO+ CmpltAbrt- UnxCmplt+ RxOF+ MalfTLP+ ECRC- UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                CEMsk:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                AERCap: First Error Pointer: 00, ECRCGenCap- ECRCGenEn- ECRCChkCap- ECRCChkEn-
                        MultHdrRecCap- MultHdrRecEn- TLPPfxPres- HdrLogCap-
                HeaderLog: 00000000 00000000 00000000 00000000
        Capabilities: [600 v1] Vendor Specific Information: ID=0001 Rev=1 Len=024 <?>
        Capabilities: [900 v1] Secondary PCI Express
                LnkCtl3: LnkEquIntrruptEn- PerformEqu-
                LaneErrStat: 0
        Capabilities: [bb0 v1] Physical Resizable BAR
                BAR 0: current size: 16MB, supported: 16MB
                BAR 1: current size: 64GB, supported: 64MB 128MB 256MB 512MB 1GB 2GB 4GB 8GB 16GB 32GB 64GB
                BAR 3: current size: 32MB, supported: 32MB
        Capabilities: [c1c v1] Physical Layer 16.0 GT/s <?>
        Capabilities: [d00 v1] Lane Margining at the Receiver <?>
        Capabilities: [e00 v1] Data Link Feature <?>
        Kernel driver in use: nvidia
        Kernel modules: nvidiafb, nouveau, nvidia_drm, nvidia
```
GPU #2
```
$ sonata@coyote2:~$ sudo lspci -vvv -s c1:00.0
c1:00.0 VGA compatible controller: NVIDIA Corporation AD102GL [RTX 6000 Ada Generation] (rev a1) (prog-if 00 [VGA controller])
        Subsystem: NVIDIA Corporation AD102GL [RTX 6000 Ada Generation]
        Control: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0
        Interrupt: pin A routed to IRQ 854
        NUMA node: 1
        Region 0: Memory at ae000000 (32-bit, non-prefetchable) [size=16M]
        Region 1: Memory at 6f000000000 (64-bit, prefetchable) [size=64G]
        Region 3: Memory at 70000000000 (64-bit, prefetchable) [size=32M]
        Region 5: I/O ports at d000 [size=128]
        Expansion ROM at af000000 [virtual] [disabled] [size=512K]
        Capabilities: [60] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=0mA PME(D0+,D1-,D2-,D3hot+,D3cold-)
                Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [68] MSI: Enable+ Count=1/1 Maskable- 64bit+
                Address: 00000000fee2e000  Data: 003b
        Capabilities: [78] Express (v2) Legacy Endpoint, MSI 00
                DevCap: MaxPayload 256 bytes, PhantFunc 0, Latency L0s unlimited, L1 <64us
                        ExtTag+ AttnBtn- AttnInd- PwrInd- RBE+ FLReset+
                DevCtl: CorrErr+ NonFatalErr+ FatalErr+ UnsupReq-
                        RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop+ FLReset-
                        MaxPayload 256 bytes, MaxReadReq 512 bytes
                DevSta: CorrErr- NonFatalErr- FatalErr- UnsupReq- AuxPwr- TransPend-
                LnkCap: Port #0, Speed 16GT/s, Width x16, ASPM L1, Exit Latency L1 <4us
                        ClockPM+ Surprise- LLActRep- BwNot- ASPMOptComp+
                LnkCtl: ASPM Disabled; RCB 64 bytes, Disabled- CommClk+
                        ExtSynch- ClockPM+ AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 2.5GT/s (downgraded), Width x16
                        TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-
                DevCap2: Completion Timeout: Range AB, TimeoutDis+ NROPrPrP- LTR+
                         10BitTagComp+ 10BitTagReq+ OBFF Via message, ExtFmt- EETLPPrefix-
                         EmergencyPowerReduction Not Supported, EmergencyPowerReductionInit-
                         FRS-
                         AtomicOpsCap: 32bit- 64bit- 128bitCAS-
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis- LTR+ 10BitTagReq- OBFF Via WAKE#,
                         AtomicOpsCtl: ReqEn-
                LnkCap2: Supported Link Speeds: 2.5-16GT/s, Crosslink- Retimer+ 2Retimers+ DRS-
                LnkCtl2: Target Link Speed: 16GT/s, EnterCompliance- SpeedDis-
                         Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
                         Compliance Preset/De-emphasis: -6dB de-emphasis, 0dB preshoot
                LnkSta2: Current De-emphasis Level: -6dB, EqualizationComplete+ EqualizationPhase1+
                         EqualizationPhase2+ EqualizationPhase3+ LinkEqualizationRequest-
                         Retimer- 2Retimers- CrosslinkRes: unsupported
        Capabilities: [b4] Vendor Specific Information: Len=14 <?>
        Capabilities: [100 v1] Virtual Channel
                Caps:   LPEVC=0 RefClk=100ns PATEntryBits=1
                Arb:    Fixed- WRR32- WRR64- WRR128-
                Ctrl:   ArbSelect=Fixed
                Status: InProgress-
                VC0:    Caps:   PATOffset=00 MaxTimeSlots=1 RejSnoopTrans-
                        Arb:    Fixed- WRR32- WRR64- WRR128- TWRR128- WRR256-
                        Ctrl:   Enable+ ID=0 ArbSelect=Fixed TC/VC=01
                        Status: NegoPending- InProgress-
        Capabilities: [250 v1] Latency Tolerance Reporting
                Max snoop latency: 34326183936ns
                Max no snoop latency: 34326183936ns
        Capabilities: [258 v1] L1 PM Substates
                L1SubCap: PCI-PM_L1.2+ PCI-PM_L1.1+ ASPM_L1.2- ASPM_L1.1+ L1_PM_Substates+
                          PortCommonModeRestoreTime=255us PortTPowerOnTime=10us
                L1SubCtl1: PCI-PM_L1.2- PCI-PM_L1.1- ASPM_L1.2- ASPM_L1.1-
                           T_CommonMode=0us
                L1SubCtl2: T_PwrOn=10us
        Capabilities: [128 v1] Power Budgeting <?>
        Capabilities: [420 v2] Advanced Error Reporting
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq+ ACSViol-
                UESvrt: DLP+ SDES+ TLP- FCP+ CmpltTO+ CmpltAbrt- UnxCmplt+ RxOF+ MalfTLP+ ECRC- UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                CEMsk:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                AERCap: First Error Pointer: 00, ECRCGenCap- ECRCGenEn- ECRCChkCap- ECRCChkEn-
                        MultHdrRecCap- MultHdrRecEn- TLPPfxPres- HdrLogCap-
                HeaderLog: 00000000 00000000 00000000 00000000
        Capabilities: [600 v1] Vendor Specific Information: ID=0001 Rev=1 Len=024 <?>
        Capabilities: [900 v1] Secondary PCI Express
                LnkCtl3: LnkEquIntrruptEn- PerformEqu-
                LaneErrStat: 0
        Capabilities: [bb0 v1] Physical Resizable BAR
                BAR 0: current size: 16MB, supported: 16MB
                BAR 1: current size: 64GB, supported: 64MB 128MB 256MB 512MB 1GB 2GB 4GB 8GB 16GB 32GB 64GB
                BAR 3: current size: 32MB, supported: 32MB
        Capabilities: [c1c v1] Physical Layer 16.0 GT/s <?>
        Capabilities: [d00 v1] Lane Margining at the Receiver <?>
        Capabilities: [e00 v1] Data Link Feature <?>
        Kernel driver in use: nvidia
        Kernel modules: nvidiafb, nouveau, nvidia_drm, nvidia
```

GPU #3
```
$ sonata@coyote2:~$ sudo lspci -vvv -s e1:00.0
e1:00.0 VGA compatible controller: NVIDIA Corporation AD102GL [RTX 6000 Ada Generation] (rev a1) (prog-if 00 [VGA controller])
        Subsystem: NVIDIA Corporation AD102GL [RTX 6000 Ada Generation]
        Control: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0
        Interrupt: pin A routed to IRQ 853
        NUMA node: 1
        Region 0: Memory at b4000000 (32-bit, non-prefetchable) [size=16M]
        Region 1: Memory at 5f000000000 (64-bit, prefetchable) [size=64G]
        Region 3: Memory at 60000000000 (64-bit, prefetchable) [size=32M]
        Region 5: I/O ports at f000 [size=128]
        Expansion ROM at b5000000 [virtual] [disabled] [size=512K]
        Capabilities: [60] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=0mA PME(D0+,D1-,D2-,D3hot+,D3cold-)
                Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [68] MSI: Enable+ Count=1/1 Maskable- 64bit+
                Address: 00000000fee2d000  Data: 003b
        Capabilities: [78] Express (v2) Legacy Endpoint, MSI 00
                DevCap: MaxPayload 256 bytes, PhantFunc 0, Latency L0s unlimited, L1 <64us
                        ExtTag+ AttnBtn- AttnInd- PwrInd- RBE+ FLReset+
                DevCtl: CorrErr+ NonFatalErr+ FatalErr+ UnsupReq-
                        RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop+ FLReset-
                        MaxPayload 256 bytes, MaxReadReq 512 bytes
                DevSta: CorrErr- NonFatalErr- FatalErr- UnsupReq- AuxPwr- TransPend-
                LnkCap: Port #0, Speed 16GT/s, Width x16, ASPM L1, Exit Latency L1 <4us
                        ClockPM+ Surprise- LLActRep- BwNot- ASPMOptComp+
                LnkCtl: ASPM Disabled; RCB 64 bytes, Disabled- CommClk+
                        ExtSynch- ClockPM+ AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 2.5GT/s (downgraded), Width x16
                        TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-
                DevCap2: Completion Timeout: Range AB, TimeoutDis+ NROPrPrP- LTR+
                         10BitTagComp+ 10BitTagReq+ OBFF Via message, ExtFmt- EETLPPrefix-
                         EmergencyPowerReduction Not Supported, EmergencyPowerReductionInit-
                         FRS-
                         AtomicOpsCap: 32bit- 64bit- 128bitCAS-
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis- LTR+ 10BitTagReq- OBFF Via WAKE#,
                         AtomicOpsCtl: ReqEn-
                LnkCap2: Supported Link Speeds: 2.5-16GT/s, Crosslink- Retimer+ 2Retimers+ DRS-
                LnkCtl2: Target Link Speed: 16GT/s, EnterCompliance- SpeedDis-
                         Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
                         Compliance Preset/De-emphasis: -6dB de-emphasis, 0dB preshoot
                LnkSta2: Current De-emphasis Level: -6dB, EqualizationComplete+ EqualizationPhase1+
                         EqualizationPhase2+ EqualizationPhase3+ LinkEqualizationRequest-
                         Retimer- 2Retimers- CrosslinkRes: unsupported
        Capabilities: [b4] Vendor Specific Information: Len=14 <?>
        Capabilities: [100 v1] Virtual Channel
                Caps:   LPEVC=0 RefClk=100ns PATEntryBits=1
                Arb:    Fixed- WRR32- WRR64- WRR128-
                Ctrl:   ArbSelect=Fixed
                Status: InProgress-
                VC0:    Caps:   PATOffset=00 MaxTimeSlots=1 RejSnoopTrans-
                        Arb:    Fixed- WRR32- WRR64- WRR128- TWRR128- WRR256-
                        Ctrl:   Enable+ ID=0 ArbSelect=Fixed TC/VC=01
                        Status: NegoPending- InProgress-
        Capabilities: [250 v1] Latency Tolerance Reporting
                Max snoop latency: 34326183936ns
                Max no snoop latency: 34326183936ns
        Capabilities: [258 v1] L1 PM Substates
                L1SubCap: PCI-PM_L1.2+ PCI-PM_L1.1+ ASPM_L1.2- ASPM_L1.1+ L1_PM_Substates+
                          PortCommonModeRestoreTime=255us PortTPowerOnTime=10us
                L1SubCtl1: PCI-PM_L1.2- PCI-PM_L1.1- ASPM_L1.2- ASPM_L1.1-
                           T_CommonMode=0us
                L1SubCtl2: T_PwrOn=10us
        Capabilities: [128 v1] Power Budgeting <?>
        Capabilities: [420 v2] Advanced Error Reporting
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq+ ACSViol-
                UESvrt: DLP+ SDES+ TLP- FCP+ CmpltTO+ CmpltAbrt- UnxCmplt+ RxOF+ MalfTLP+ ECRC- UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                CEMsk:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                AERCap: First Error Pointer: 00, ECRCGenCap- ECRCGenEn- ECRCChkCap- ECRCChkEn-
                        MultHdrRecCap- MultHdrRecEn- TLPPfxPres- HdrLogCap-
                HeaderLog: 00000000 00000000 00000000 00000000
        Capabilities: [600 v1] Vendor Specific Information: ID=0001 Rev=1 Len=024 <?>
        Capabilities: [900 v1] Secondary PCI Express
                LnkCtl3: LnkEquIntrruptEn- PerformEqu-
                LaneErrStat: 0
        Capabilities: [bb0 v1] Physical Resizable BAR
                BAR 0: current size: 16MB, supported: 16MB
                BAR 1: current size: 64GB, supported: 64MB 128MB 256MB 512MB 1GB 2GB 4GB 8GB 16GB 32GB 64GB
                BAR 3: current size: 32MB, supported: 32MB
        Capabilities: [c1c v1] Physical Layer 16.0 GT/s <?>
        Capabilities: [d00 v1] Lane Margining at the Receiver <?>
        Capabilities: [e00 v1] Data Link Feature <?>
        Kernel driver in use: nvidia
        Kernel modules: nvidiafb, nouveau, nvidia_drm, nvidia
```

ConnectX-7 #0
```
$ sonata@coyote2:~$ sudo lspci -vvv -s 21:00.0
21:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
        Subsystem: Mellanox Technologies MT2910 Family [ConnectX-7]
        Control: I/O- Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0, Cache Line Size: 64 bytes
        Interrupt: pin A routed to IRQ 55
        NUMA node: 0
        Region 0: Memory at 500aa000000 (64-bit, prefetchable) [size=32M]
        Expansion ROM at b8100000 [disabled] [size=1M]
        Capabilities: [60] Express (v2) Endpoint, MSI 00
                DevCap: MaxPayload 512 bytes, PhantFunc 0, Latency L0s unlimited, L1 unlimited
                        ExtTag+ AttnBtn- AttnInd- PwrInd- RBE+ FLReset+ SlotPowerLimit 75W
                DevCtl: CorrErr+ NonFatalErr+ FatalErr+ UnsupReq-
                        RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop+ FLReset-
                        MaxPayload 512 bytes, MaxReadReq 512 bytes
                DevSta: CorrErr+ NonFatalErr- FatalErr- UnsupReq+ AuxPwr- TransPend-
                LnkCap: Port #0, Speed 32GT/s, Width x16, ASPM not supported
                        ClockPM- Surprise- LLActRep- BwNot- ASPMOptComp+
                LnkCtl: ASPM Disabled; RCB 64 bytes, Disabled- CommClk+
                        ExtSynch- ClockPM- AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 32GT/s, Width x16
                        TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-
                DevCap2: Completion Timeout: Range ABC, TimeoutDis+ NROPrPrP- LTR-
                         10BitTagComp+ 10BitTagReq+ OBFF Not Supported, ExtFmt- EETLPPrefix-
                         EmergencyPowerReduction Not Supported, EmergencyPowerReductionInit-
                         FRS- TPHComp- ExtTPHComp-
                         AtomicOpsCap: 32bit+ 64bit+ 128bitCAS+
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis- LTR- 10BitTagReq- OBFF Disabled,
                         AtomicOpsCtl: ReqEn+
                LnkCap2: Supported Link Speeds: 2.5-32GT/s, Crosslink- Retimer+ 2Retimers+ DRS-
                LnkCtl2: Target Link Speed: 32GT/s, EnterCompliance- SpeedDis-
                         Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
                         Compliance Preset/De-emphasis: -6dB de-emphasis, 0dB preshoot
                LnkSta2: Current De-emphasis Level: -6dB, EqualizationComplete+ EqualizationPhase1+
                         EqualizationPhase2+ EqualizationPhase3+ LinkEqualizationRequest-
                         Retimer- 2Retimers- CrosslinkRes: unsupported
        Capabilities: [48] Vital Product Data
                Product Name: NVIDIA ConnectX-7 Adapter Card, 200GbE / NDR200, Dual-port QSFP112, PCIe 5.0x16 with x16 PCIe extension option, Crypto Disabled, Secure Boot Enabled
                Read-only fields:
                        [PN] Part number: MCX755106AS-HEAT
                        [EC] Engineering changes: A8
                        [V2] Vendor specific: MCX755106AS-HEAT
                        [SN] Serial number: MT24216006W0
                        [V3] Vendor specific: 74173d198917ef1180009c63c0d420f0
                        [VA] Vendor specific: MLX:MN=MLNX:CSKU=V2:UUID=V3:PCI=V0:MODL=CX755106A
                        [V0] Vendor specific: PCIeGen5 x16
                        [VU] Vendor specific: MT24216006W0MLNXS0D0F0
                        [RV] Reserved: checksum good, 1 byte(s) reserved
                End
        Capabilities: [9c] MSI-X: Enable+ Count=64 Masked-
                Vector table: BAR=0 offset=00002000
                PBA: BAR=0 offset=00003000
        Capabilities: [c0] Vendor Specific Information: Len=18 <?>
        Capabilities: [40] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=375mA PME(D0-,D1-,D2-,D3hot-,D3cold+)
                Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [100 v1] Advanced Error Reporting
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq+ ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq+ ACSViol-
                UESvrt: DLP+ SDES- TLP- FCP+ CmpltTO+ CmpltAbrt- UnxCmplt+ RxOF+ MalfTLP+ ECRC+ UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr+
                CEMsk:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                AERCap: First Error Pointer: 08, ECRCGenCap+ ECRCGenEn- ECRCChkCap+ ECRCChkEn-
                        MultHdrRecCap- MultHdrRecEn- TLPPfxPres- HdrLogCap-
                HeaderLog: 00000000 00000000 00000000 00000000
        Capabilities: [150 v1] Alternative Routing-ID Interpretation (ARI)
                ARICap: MFVC- ACS-, Next Function: 1
                ARICtl: MFVC- ACS-, Function Group: 0
        Capabilities: [180 v1] Single Root I/O Virtualization (SR-IOV)
                IOVCap: Migration- 10BitTagReq- Interrupt Message Number: 000
                IOVCtl: Enable- Migration- Interrupt- MSE- ARIHierarchy+ 10BitTagReq-
                IOVSta: Migration-
                Initial VFs: 16, Total VFs: 16, Number of VFs: 0, Function Dependency Link: 00
                VF offset: 2, stride: 1, Device ID: 101e
                Supported Page Size: 000007ff, System Page Size: 00000001
                Region 0: Memory at 00000500ae000000 (64-bit, prefetchable)
                VF Migration: offset: 00000000, BIR: 0
        Capabilities: [1c0 v1] Secondary PCI Express
                LnkCtl3: LnkEquIntrruptEn- PerformEqu-
                LaneErrStat: 0
        Capabilities: [230 v1] Access Control Services
                ACSCap: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
        Capabilities: [320 v1] Lane Margining at the Receiver <?>
        Capabilities: [370 v1] Physical Layer 16.0 GT/s <?>
        Capabilities: [3b0 v1] Extended Capability ID 0x2a
        Capabilities: [420 v1] Data Link Feature <?>
        Kernel driver in use: mlx5_core
        Kernel modules: mlx5_core
```

ConnectX-7 #1
```
$ sonata@coyote2:~$ sudo lspci -vvv -s 81:00.0
81:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
        Subsystem: Mellanox Technologies MT2910 Family [ConnectX-7]
        Control: I/O- Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0, Cache Line Size: 64 bytes
        Interrupt: pin A routed to IRQ 712
        NUMA node: 1
        Region 0: Memory at 8013a000000 (64-bit, prefetchable) [size=32M]
        Expansion ROM at abe00000 [disabled] [size=1M]
        Capabilities: [60] Express (v2) Endpoint, MSI 00
                DevCap: MaxPayload 512 bytes, PhantFunc 0, Latency L0s unlimited, L1 unlimited
                        ExtTag+ AttnBtn- AttnInd- PwrInd- RBE+ FLReset+ SlotPowerLimit 75W
                DevCtl: CorrErr+ NonFatalErr+ FatalErr+ UnsupReq-
                        RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop+ FLReset-
                        MaxPayload 512 bytes, MaxReadReq 512 bytes
                DevSta: CorrErr+ NonFatalErr- FatalErr- UnsupReq+ AuxPwr- TransPend-
                LnkCap: Port #0, Speed 32GT/s, Width x16, ASPM not supported
                        ClockPM- Surprise- LLActRep- BwNot- ASPMOptComp+
                LnkCtl: ASPM Disabled; RCB 64 bytes, Disabled- CommClk+
                        ExtSynch- ClockPM- AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 32GT/s, Width x16
                        TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-
                DevCap2: Completion Timeout: Range ABC, TimeoutDis+ NROPrPrP- LTR-
                         10BitTagComp+ 10BitTagReq+ OBFF Not Supported, ExtFmt- EETLPPrefix-
                         EmergencyPowerReduction Not Supported, EmergencyPowerReductionInit-
                         FRS- TPHComp- ExtTPHComp-
                         AtomicOpsCap: 32bit+ 64bit+ 128bitCAS+
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis- LTR- 10BitTagReq- OBFF Disabled,
                         AtomicOpsCtl: ReqEn+
                LnkCap2: Supported Link Speeds: 2.5-32GT/s, Crosslink- Retimer+ 2Retimers+ DRS-
                LnkCtl2: Target Link Speed: 32GT/s, EnterCompliance- SpeedDis-
                         Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
                         Compliance Preset/De-emphasis: -6dB de-emphasis, 0dB preshoot
                LnkSta2: Current De-emphasis Level: -6dB, EqualizationComplete+ EqualizationPhase1+
                         EqualizationPhase2+ EqualizationPhase3+ LinkEqualizationRequest-
                         Retimer- 2Retimers- CrosslinkRes: unsupported
        Capabilities: [48] Vital Product Data
                Product Name: NVIDIA ConnectX-7 Adapter Card, 200GbE / NDR200, Dual-port QSFP112, PCIe 5.0x16 with x16 PCIe extension option, Crypto Disabled, Secure Boot Enabled
                Read-only fields:
                        [PN] Part number: MCX755106AS-HEAT
                        [EC] Engineering changes: A8
                        [V2] Vendor specific: MCX755106AS-HEAT
                        [SN] Serial number: MT24426016T3
                        [V3] Vendor specific: 227c69922c8eef118000b8e9244edeb0
                        [VA] Vendor specific: MLX:MN=MLNX:CSKU=V2:UUID=V3:PCI=V0:MODL=CX755106A
                        [V0] Vendor specific: PCIeGen5 x16
                        [VU] Vendor specific: MT24426016T3MLNXS0D0F0
                        [RV] Reserved: checksum good, 1 byte(s) reserved
                End
        Capabilities: [9c] MSI-X: Enable+ Count=64 Masked-
                Vector table: BAR=0 offset=00002000
                PBA: BAR=0 offset=00003000
        Capabilities: [c0] Vendor Specific Information: Len=18 <?>
        Capabilities: [40] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=375mA PME(D0-,D1-,D2-,D3hot-,D3cold+)
                Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [100 v1] Advanced Error Reporting
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq+ ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq+ ACSViol-
                UESvrt: DLP+ SDES- TLP- FCP+ CmpltTO+ CmpltAbrt- UnxCmplt+ RxOF+ MalfTLP+ ECRC+ UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr+
                CEMsk:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                AERCap: First Error Pointer: 08, ECRCGenCap+ ECRCGenEn- ECRCChkCap+ ECRCChkEn-
                        MultHdrRecCap- MultHdrRecEn- TLPPfxPres- HdrLogCap-
                HeaderLog: 00000000 00000000 00000000 00000000
        Capabilities: [150 v1] Alternative Routing-ID Interpretation (ARI)
                ARICap: MFVC- ACS-, Next Function: 1
                ARICtl: MFVC- ACS-, Function Group: 0
        Capabilities: [180 v1] Single Root I/O Virtualization (SR-IOV)
                IOVCap: Migration- 10BitTagReq- Interrupt Message Number: 000
                IOVCtl: Enable- Migration- Interrupt- MSE- ARIHierarchy+ 10BitTagReq-
                IOVSta: Migration-
                Initial VFs: 16, Total VFs: 16, Number of VFs: 0, Function Dependency Link: 00
                VF offset: 2, stride: 1, Device ID: 101e
                Supported Page Size: 000007ff, System Page Size: 00000001
                Region 0: Memory at 000008013e000000 (64-bit, prefetchable)
                VF Migration: offset: 00000000, BIR: 0
        Capabilities: [1c0 v1] Secondary PCI Express
                LnkCtl3: LnkEquIntrruptEn- PerformEqu-
                LaneErrStat: 0
        Capabilities: [230 v1] Access Control Services
                ACSCap: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
        Capabilities: [240 v1] Precision Time Measurement
                PTMCap: Requester:+ Responder:- Root:-
                PTMClockGranularity: Unimplemented
                PTMControl: Enabled:- RootSelected:-
                PTMEffectiveGranularity: Unknown
        Capabilities: [320 v1] Lane Margining at the Receiver <?>
        Capabilities: [370 v1] Physical Layer 16.0 GT/s <?>
        Capabilities: [3b0 v1] Extended Capability ID 0x2a
        Capabilities: [420 v1] Data Link Feature <?>
        Kernel driver in use: mlx5_core
        Kernel modules: mlx5_core
```

HighPoint Carrier Board #0
```
$ sonata@coyote2:~$ sudo lspci -vvv -s 71:00.0
71:00.0 RAID bus controller: HighPoint Technologies, Inc. SSD7540 PCIe Gen4 x16 8-Port M.2 NVMe RAID Controller (rev 01)
        Subsystem: HighPoint Technologies, Inc. SSD7540 PCIe Gen4 x16 8-Port M.2 NVMe RAID Controller
        Control: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx-
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0, Cache Line Size: 64 bytes
        Interrupt: pin A routed to IRQ 255
        NUMA node: 0
        Region 0: I/O ports at 7050 [size=8]
        Region 1: I/O ports at 7040 [size=4]
        Region 2: I/O ports at 7030 [size=8]
        Region 3: I/O ports at 7020 [size=4]
        Region 4: I/O ports at 7000 [size=32]
        Region 5: Memory at c6810000 (32-bit, non-prefetchable) [size=2K]
        Expansion ROM at c6800000 [disabled] [size=64K]
        Capabilities: [40] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=0mA PME(D0-,D1-,D2-,D3hot+,D3cold-)
                Status: D0 NoSoftRst- PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [50] MSI: Enable- Count=1/1 Maskable- 64bit-
                Address: 00000000  Data: 0000
        Capabilities: [70] Express (v2) Legacy Endpoint, MSI 00
                DevCap: MaxPayload 512 bytes, PhantFunc 0, Latency L0s <1us, L1 <8us
                        ExtTag- AttnBtn- AttnInd- PwrInd- RBE+ FLReset-
                DevCtl: CorrErr+ NonFatalErr+ FatalErr+ UnsupReq-
                        RlxdOrd+ ExtTag- PhantFunc- AuxPwr- NoSnoop-
                        MaxPayload 256 bytes, MaxReadReq 512 bytes
                DevSta: CorrErr- NonFatalErr- FatalErr- UnsupReq- AuxPwr- TransPend-
                LnkCap: Port #0, Speed 5GT/s, Width x2, ASPM L0s L1, Exit Latency L0s <512ns, L1 <64us
                        ClockPM- Surprise- LLActRep- BwNot- ASPMOptComp-
                LnkCtl: ASPM Disabled; RCB 64 bytes, Disabled- CommClk-
                        ExtSynch- ClockPM- AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 5GT/s, Width x1 (downgraded)
                        TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-
                DevCap2: Completion Timeout: Not Supported, TimeoutDis+ NROPrPrP- LTR-
                         10BitTagComp- 10BitTagReq- OBFF Not Supported, ExtFmt- EETLPPrefix-
                         EmergencyPowerReduction Not Supported, EmergencyPowerReductionInit-
                         FRS-
                         AtomicOpsCap: 32bit- 64bit- 128bitCAS-
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis- LTR- 10BitTagReq- OBFF Disabled,
                         AtomicOpsCtl: ReqEn-
                LnkCtl2: Target Link Speed: 5GT/s, EnterCompliance- SpeedDis-
                         Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
                         Compliance Preset/De-emphasis: -6dB de-emphasis, 0dB preshoot
                LnkSta2: Current De-emphasis Level: -6dB, EqualizationComplete- EqualizationPhase1-
                         EqualizationPhase2- EqualizationPhase3- LinkEqualizationRequest-
                         Retimer- 2Retimers- CrosslinkRes: unsupported
        Capabilities: [e0] SATA HBA v0.0 BAR4 Offset=00000004
        Capabilities: [100 v1] Advanced Error Reporting
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq+ ACSViol-
                UESvrt: DLP+ SDES+ TLP- FCP+ CmpltTO+ CmpltAbrt- UnxCmplt+ RxOF+ MalfTLP+ ECRC+ UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                CEMsk:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                AERCap: First Error Pointer: 00, ECRCGenCap- ECRCGenEn- ECRCChkCap- ECRCChkEn-
                        MultHdrRecCap- MultHdrRecEn- TLPPfxPres- HdrLogCap-
                HeaderLog: 00000000 00000000 00000000 00000000
```

HighPoint Carrier Board #1
```
$ sonata@coyote2:~$ sudo lspci -vvv -s b1:00.0
b1:00.0 RAID bus controller: HighPoint Technologies, Inc. SSD7540 PCIe Gen4 x16 8-Port M.2 NVMe RAID Controller (rev 01)
        Subsystem: HighPoint Technologies, Inc. SSD7540 PCIe Gen4 x16 8-Port M.2 NVMe RAID Controller
        Control: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx-
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0, Cache Line Size: 64 bytes
        Interrupt: pin A routed to IRQ 255
        NUMA node: 1
        Region 0: I/O ports at b050 [size=8]
        Region 1: I/O ports at b040 [size=4]
        Region 2: I/O ports at b030 [size=8]
        Region 3: I/O ports at b020 [size=4]
        Region 4: I/O ports at b000 [size=32]
        Region 5: Memory at a3810000 (32-bit, non-prefetchable) [size=2K]
        Expansion ROM at a3800000 [disabled] [size=64K]
        Capabilities: [40] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=0mA PME(D0-,D1-,D2-,D3hot+,D3cold-)
                Status: D0 NoSoftRst- PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [50] MSI: Enable- Count=1/1 Maskable- 64bit-
                Address: 00000000  Data: 0000
        Capabilities: [70] Express (v2) Legacy Endpoint, MSI 00
                DevCap: MaxPayload 512 bytes, PhantFunc 0, Latency L0s <1us, L1 <8us
                        ExtTag- AttnBtn- AttnInd- PwrInd- RBE+ FLReset-
                DevCtl: CorrErr+ NonFatalErr+ FatalErr+ UnsupReq-
                        RlxdOrd+ ExtTag- PhantFunc- AuxPwr- NoSnoop-
                        MaxPayload 256 bytes, MaxReadReq 512 bytes
                DevSta: CorrErr- NonFatalErr- FatalErr- UnsupReq- AuxPwr- TransPend-
                LnkCap: Port #0, Speed 5GT/s, Width x2, ASPM L0s L1, Exit Latency L0s <512ns, L1 <64us
                        ClockPM- Surprise- LLActRep- BwNot- ASPMOptComp-
                LnkCtl: ASPM Disabled; RCB 64 bytes, Disabled- CommClk-
                        ExtSynch- ClockPM- AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 5GT/s, Width x1 (downgraded)
                        TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-
                DevCap2: Completion Timeout: Not Supported, TimeoutDis+ NROPrPrP- LTR-
                         10BitTagComp- 10BitTagReq- OBFF Not Supported, ExtFmt- EETLPPrefix-
                         EmergencyPowerReduction Not Supported, EmergencyPowerReductionInit-
                         FRS-
                         AtomicOpsCap: 32bit- 64bit- 128bitCAS-
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis- LTR- 10BitTagReq- OBFF Disabled,
                         AtomicOpsCtl: ReqEn-
                LnkCtl2: Target Link Speed: 5GT/s, EnterCompliance- SpeedDis-
                         Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
                         Compliance Preset/De-emphasis: -6dB de-emphasis, 0dB preshoot
                LnkSta2: Current De-emphasis Level: -6dB, EqualizationComplete- EqualizationPhase1-
                         EqualizationPhase2- EqualizationPhase3- LinkEqualizationRequest-
                         Retimer- 2Retimers- CrosslinkRes: unsupported
        Capabilities: [e0] SATA HBA v0.0 BAR4 Offset=00000004
        Capabilities: [100 v1] Advanced Error Reporting
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq+ ACSViol-
                UESvrt: DLP+ SDES+ TLP- FCP+ CmpltTO+ CmpltAbrt- UnxCmplt+ RxOF+ MalfTLP+ ECRC+ UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                CEMsk:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                AERCap: First Error Pointer: 00, ECRCGenCap- ECRCGenEn- ECRCChkCap- ECRCChkEn-
                        MultHdrRecCap- MultHdrRecEn- TLPPfxPres- HdrLogCap-
                HeaderLog: 00000000 00000000 00000000 00000000
```

## NVIDIA SMI Debug
```
$ sonata@coyote2:~$ nvidia-smi
Tue Jul 22 20:50:11 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.51.03              Driver Version: 575.51.03      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:01:00.0 Off |                  Off |
| 30%   31C    P8              5W /  300W |       2MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:41:00.0 Off |                  Off |
| 30%   34C    P8             11W /  300W |       2MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:C1:00.0 Off |                  Off |
| 30%   30C    P8              7W /  300W |       2MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:E1:00.0 Off |                  Off |
| 30%   32C    P8              5W /  300W |       2MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

```
$ sonata@coyote2:~$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    NIC0    NIC1    NIC2    NIC3    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NODE    SYS     SYS     NODE    NODE    SYS     SYS     0-31    0               N/A
GPU1    NODE     X      SYS     SYS     NODE    NODE    SYS     SYS     0-31    0               N/A
GPU2    SYS     SYS      X      NODE    SYS     SYS     NODE    NODE    32-63   1               N/A
GPU3    SYS     SYS     NODE     X      SYS     SYS     NODE    NODE    32-63   1               N/A
NIC0    NODE    NODE    SYS     SYS      X      PIX     SYS     SYS
NIC1    NODE    NODE    SYS     SYS     PIX      X      SYS     SYS
NIC2    SYS     SYS     NODE    NODE    SYS     SYS      X      PIX
NIC3    SYS     SYS     NODE    NODE    SYS     SYS     PIX      X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
```

## NVIDIA Transfers
```
$ sonata@coyote2:~/cuda-samples/Samples/1_Utilities/bandwidthTest$ ./bandwidthTest --device=0
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: NVIDIA RTX 6000 Ada Generation
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     26.8

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     26.3

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     4366.8

Result = PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

$ sonata@coyote2:~/cuda-samples/Samples/1_Utilities/bandwidthTest$ ./bandwidthTest --device=1
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 1: NVIDIA RTX 6000 Ada Generation
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     26.8

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     26.3

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     4368.3

Result = PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

$ sonata@coyote2:~/cuda-samples/Samples/1_Utilities/bandwidthTest$ ./bandwidthTest --device=2
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 2: NVIDIA RTX 6000 Ada Generation
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     26.8

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     26.3

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     4393.3

Result = PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

$ sonata@coyote2:~/cuda-samples/Samples/1_Utilities/bandwidthTest$ ./bandwidthTest --device=3
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 3: NVIDIA RTX 6000 Ada Generation
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     26.8

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     26.3

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(GB/s)
   32000000                     4301.6

Result = PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
```

```
$ sonata@coyote2:~/cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest$ ./p2pBandwidthLatencyTest
[P2P (Peer-to-Peer) GPU Bandwidth Latency Test]
Device: 0, NVIDIA RTX 6000 Ada Generation, pciBusID: 1, pciDeviceID: 0, pciDomainID:0
Device: 1, NVIDIA RTX 6000 Ada Generation, pciBusID: 41, pciDeviceID: 0, pciDomainID:0
Device: 2, NVIDIA RTX 6000 Ada Generation, pciBusID: c1, pciDeviceID: 0, pciDomainID:0
Device: 3, NVIDIA RTX 6000 Ada Generation, pciBusID: e1, pciDeviceID: 0, pciDomainID:0
Device=0 CAN Access Peer Device=1
Device=0 CAN Access Peer Device=2
Device=0 CAN Access Peer Device=3
Device=1 CAN Access Peer Device=0
Device=1 CAN Access Peer Device=2
Device=1 CAN Access Peer Device=3
Device=2 CAN Access Peer Device=0
Device=2 CAN Access Peer Device=1
Device=2 CAN Access Peer Device=3
Device=3 CAN Access Peer Device=0
Device=3 CAN Access Peer Device=1
Device=3 CAN Access Peer Device=2

***NOTE: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.
So you can see lesser Bandwidth (GB/s) and unstable Latency (us) in those cases.

P2P Connectivity Matrix
     D\D     0     1     2     3
     0       1     1     1     1
     1       1     1     1     1
     2       1     1     1     1
     3       1     1     1     1
Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3
     0 140.75  22.23  22.12  22.21
     1  22.32 813.80  22.14  22.38
     2  22.35  22.17 812.53  22.10
     3  22.23  22.24  22.25 813.80
Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   D\D     0      1      2      3
     0 802.93  26.33  26.32  26.32
     1  26.32 833.33  26.32  26.32
     2  26.32  26.32 830.69  26.32
     3  26.33  26.33  26.33 829.35
Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3
     0 800.04  31.53  31.45  31.66
     1  31.52 812.06  31.72  31.66
     2  31.44  31.51 810.01  31.59
     3  31.39  31.54  31.57 810.22
Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3
     0 804.38  52.10  52.10  52.11
     1  52.10 805.20  52.12  52.11
     2  52.12  52.10 803.76  52.12
     3  52.11  52.11  52.10 803.34
P2P=Disabled Latency Matrix (us)
   GPU     0      1      2      3
     0   1.37  10.73  13.76  13.64
     1  10.27   1.37  14.08  14.22
     2  11.87  11.58   1.39  11.50
     3  11.87  11.77  11.60   1.39

   CPU     0      1      2      3
     0   1.86   5.27   6.01   6.04
     1   5.24   1.73   6.00   5.96
     2   5.67   5.67   2.03   6.53
     3   5.75   5.66   6.48   2.04
P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1      2      3
     0   1.36   1.24   1.34   1.32
     1   1.21   1.37   1.29   1.32
     2   1.42   1.40   1.35   1.34
     3   1.44   1.43   1.35   1.40

   CPU     0      1      2      3
     0   1.78   1.43   1.43   1.41
     1   1.53   1.83   1.43   1.46
     2   1.72   1.68   2.07   1.68
     3   1.74   1.71   1.73   2.05

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
```

## RDMA P2P Transfers
```
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-0/test.bin -x 0 -d 0 -s 50G -I 1 -i 1G
IoType: WRITE XferType: GPUD Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 22.016713 GiB/sec, Avg_Latency: 44284.240000 usecs ops: 50 total_time 2.271002 secs
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-0/test.bin -x 0 -d 1 -s 50G -I 1 -i 1G
IoType: WRITE XferType: GPUD Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 22.816194 GiB/sec, Avg_Latency: 42468.840000 usecs ops: 50 total_time 2.191426 secs
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-0/test.bin -x 0 -d 2 -s 50G -I 1 -i 1G
IoType: WRITE XferType: GPUD Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 23.723779 GiB/sec, Avg_Latency: 40707.720000 usecs ops: 50 total_time 2.107590 secs
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-0/test.bin -x 0 -d 3 -s 50G -I 1 -i 1G
IoType: WRITE XferType: GPUD Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 23.561489 GiB/sec, Avg_Latency: 41206.740000 usecs ops: 50 total_time 2.122107 secs
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-1/test.bin -x 0 -d 0 -s 50G -I 1 -i 1G
IoType: WRITE XferType: GPUD Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 22.874219 GiB/sec, Avg_Latency: 42452.880000 usecs ops: 50 total_time 2.185867 secs
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-1/test.bin -x 0 -d 1 -s 50G -I 1 -i 1G
IoType: WRITE XferType: GPUD Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 23.673222 GiB/sec, Avg_Latency: 41014.720000 usecs ops: 50 total_time 2.112091 secs
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-1/test.bin -x 0 -d 2 -s 50G -I 1 -i 1G
IoType: WRITE XferType: GPUD Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 24.068362 GiB/sec, Avg_Latency: 40345.180000 usecs ops: 50 total_time 2.077416 secs
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-1/test.bin -x 0 -d 3 -s 50G -I 1 -i 1G
IoType: WRITE XferType: GPUD Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 24.352646 GiB/sec, Avg_Latency: 39787.980000 usecs ops: 50 total_time 2.053165 secs
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-0/test.bin -x 1 -d 0 -s 50G -I 1 -i 1G
IoType: WRITE XferType: CPUONLY Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 24.961796 GiB/sec, Avg_Latency: 39117.240000 usecs ops: 50 total_time 2.003061 secs
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-0/test.bin -x 1 -d 1 -s 50G -I 1 -i 1G
IoType: WRITE XferType: CPUONLY Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 25.474384 GiB/sec, Avg_Latency: 38311.280000 usecs ops: 50 total_time 1.962756 secs
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-0/test.bin -x 1 -d 2 -s 50G -I 1 -i 1G
IoType: WRITE XferType: CPUONLY Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 25.343018 GiB/sec, Avg_Latency: 38515.040000 usecs ops: 50 total_time 1.972930 secs
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-0/test.bin -x 1 -d 3 -s 50G -I 1 -i 1G
IoType: WRITE XferType: CPUONLY Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 25.394968 GiB/sec, Avg_Latency: 38433.720000 usecs ops: 50 total_time 1.968894 secs
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-1/test.bin -x 1 -d 0 -s 50G -I 1 -i 1G
IoType: WRITE XferType: CPUONLY Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 24.974875 GiB/sec, Avg_Latency: 39095.660000 usecs ops: 50 total_time 2.002012 secs
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-1/test.bin -x 1 -d 1 -s 50G -I 1 -i 1G
IoType: WRITE XferType: CPUONLY Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 24.539901 GiB/sec, Avg_Latency: 39804.800000 usecs ops: 50 total_time 2.037498 secs
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-1/test.bin -x 1 -d 2 -s 50G -I 1 -i 1G
IoType: WRITE XferType: CPUONLY Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 25.056176 GiB/sec, Avg_Latency: 38965.000000 usecs ops: 50 total_time 1.995516 secs
sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-1/test.bin -x 1 -d 3 -s 50G -I 1 -i 1G
IoType: WRITE XferType: CPUONLY Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 24.802731 GiB/sec, Avg_Latency: 39372.900000 usecs ops: 50 total_time 2.015907 secs
$ sonata@coyote2:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid-1/test.bin -x 1 -d 4 -s 50G -I 1 -i 1G
IoType: WRITE XferType: CPUONLY Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 25.237829 GiB/sec, Avg_Latency: 38678.640000 usecs ops: 50 total_time 1.981153 secs
```

## ACS State
```
$ sonata@coyote2:~$ sudo lspci -vvv | grep ACSCtl
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
```

## CPU Topology
```
$ sonata@coyote2:~$ lscpu
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          52 bits physical, 57 bits virtual
  Byte Order:             Little Endian
CPU(s):                   64
  On-line CPU(s) list:    0-63
Vendor ID:                AuthenticAMD
  Model name:             AMD EPYC 9374F 32-Core Processor
    CPU family:           25
    Model:                17
    Thread(s) per core:   1
    Core(s) per socket:   32
    Socket(s):            2
    Stepping:             1
    Frequency boost:      enabled
    CPU(s) scaling MHz:   45%
    CPU max MHz:          3850.0000
    CPU min MHz:          1500.0000
    BogoMIPS:             7687.93
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good a
                          md_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_l
                          egacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_ps
                          tate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt
                           clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr
                          rdpru wbnoinvd amd_ppin cppc amd_ibpb_ret arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vg
                          if x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca f
                          srm flush_l1d debug_swap
Virtualization features:
  Virtualization:         AMD-V
Caches (sum of all):
  L1d:                    2 MiB (64 instances)
  L1i:                    2 MiB (64 instances)
  L2:                     64 MiB (64 instances)
  L3:                     512 MiB (16 instances)
NUMA:
  NUMA node(s):           2
  NUMA node0 CPU(s):      0-31
  NUMA node1 CPU(s):      32-63
Vulnerabilities:
  Gather data sampling:   Not affected
  Itlb multihit:          Not affected
  L1tf:                   Not affected
  Mds:                    Not affected
  Meltdown:               Not affected
  Mmio stale data:        Not affected
  Reg file data sampling: Not affected
  Retbleed:               Not affected
  Spec rstack overflow:   Mitigation; Safe RET
  Spec store bypass:      Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:             Mitigation; Enhanced / Automatic IBRS; IBPB conditional; STIBP disabled; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
  Srbds:                  Not affected
  Tsx async abort:        Not affected
```

```
$ sonata@coyote2:~$ lscpu -e
CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ    MINMHZ       MHZ
  0    0      0    0 0:0:0:0          yes 3850.0000 1500.0000 3850.0000
  1    0      0    1 1:1:1:0          yes 3850.0000 1500.0000 1500.0000
  2    0      0    2 2:2:2:0          yes 3850.0000 1500.0000 2600.0000
  3    0      0    3 3:3:3:0          yes 3850.0000 1500.0000 2600.0000
  4    0      0    4 16:16:16:4       yes 3850.0000 1500.0000 1496.7360
  5    0      0    5 17:17:17:4       yes 3850.0000 1500.0000 1500.0000
  6    0      0    6 18:18:18:4       yes 3850.0000 1500.0000 1500.0000
  7    0      0    7 19:19:19:4       yes 3850.0000 1500.0000 1500.0000
  8    0      0    8 8:8:8:2          yes 3850.0000 1500.0000 2595.9131
  9    0      0    9 9:9:9:2          yes 3850.0000 1500.0000 1500.0000
 10    0      0   10 10:10:10:2       yes 3850.0000 1500.0000 1500.0000
 11    0      0   11 11:11:11:2       yes 3850.0000 1500.0000 1500.0000
 12    0      0   12 24:24:24:6       yes 3850.0000 1500.0000 1500.0000
 13    0      0   13 25:25:25:6       yes 3850.0000 1500.0000 1500.0000
 14    0      0   14 26:26:26:6       yes 3850.0000 1500.0000 1500.0000
 15    0      0   15 27:27:27:6       yes 3850.0000 1500.0000 1500.0000
 16    0      0   16 12:12:12:3       yes 3850.0000 1500.0000 1500.0000
 17    0      0   17 13:13:13:3       yes 3850.0000 1500.0000 1500.0000
 18    0      0   18 14:14:14:3       yes 3850.0000 1500.0000 1500.0000
 19    0      0   19 15:15:15:3       yes 3850.0000 1500.0000 1500.0000
 20    0      0   20 28:28:28:7       yes 3850.0000 1500.0000 1500.0000
 21    0      0   21 29:29:29:7       yes 3850.0000 1500.0000 1500.0000
 22    0      0   22 30:30:30:7       yes 3850.0000 1500.0000 1500.0000
 23    0      0   23 31:31:31:7       yes 3850.0000 1500.0000 1500.0000
 24    0      0   24 4:4:4:1          yes 3850.0000 1500.0000 1500.0000
 25    0      0   25 5:5:5:1          yes 3850.0000 1500.0000 1500.0000
 26    0      0   26 6:6:6:1          yes 3850.0000 1500.0000 1500.0000
 27    0      0   27 7:7:7:1          yes 3850.0000 1500.0000 1500.0000
 28    0      0   28 20:20:20:5       yes 3850.0000 1500.0000 1500.0000
 29    0      0   29 21:21:21:5       yes 3850.0000 1500.0000 1500.0000
 30    0      0   30 22:22:22:5       yes 3850.0000 1500.0000 1500.0000
 31    0      0   31 23:23:23:5       yes 3850.0000 1500.0000 1500.0000
 32    1      1   32 32:32:32:8       yes 3850.0000 1500.0000 1500.0000
 33    1      1   33 33:33:33:8       yes 3850.0000 1500.0000 2600.0000
 34    1      1   34 34:34:34:8       yes 3850.0000 1500.0000 2600.0000
 35    1      1   35 35:35:35:8       yes 3850.0000 1500.0000 2600.0000
 36    1      1   36 48:48:48:12      yes 3850.0000 1500.0000 1500.0000
 37    1      1   37 49:49:49:12      yes 3850.0000 1500.0000 1496.9091
 38    1      1   38 50:50:50:12      yes 3850.0000 1500.0000 1500.0000
 39    1      1   39 51:51:51:12      yes 3850.0000 1500.0000 1500.0000
 40    1      1   40 40:40:40:10      yes 3850.0000 1500.0000 1500.0000
 41    1      1   41 41:41:41:10      yes 3850.0000 1500.0000 1500.0000
 42    1      1   42 42:42:42:10      yes 3850.0000 1500.0000 1500.0000
 43    1      1   43 43:43:43:10      yes 3850.0000 1500.0000 1500.0000
 44    1      1   44 56:56:56:14      yes 3850.0000 1500.0000 1500.0000
 45    1      1   45 57:57:57:14      yes 3850.0000 1500.0000 3850.0000
 46    1      1   46 58:58:58:14      yes 3850.0000 1500.0000 1500.0000
 47    1      1   47 59:59:59:14      yes 3850.0000 1500.0000 3850.0000
 48    1      1   48 44:44:44:11      yes 3850.0000 1500.0000 1500.0000
 49    1      1   49 45:45:45:11      yes 3850.0000 1500.0000 1500.0000
 50    1      1   50 46:46:46:11      yes 3850.0000 1500.0000 1500.0000
 51    1      1   51 47:47:47:11      yes 3850.0000 1500.0000 1500.0000
 52    1      1   52 60:60:60:15      yes 3850.0000 1500.0000 1500.0000
 53    1      1   53 61:61:61:15      yes 3850.0000 1500.0000 1500.0000
 54    1      1   54 62:62:62:15      yes 3850.0000 1500.0000 1500.0000
 55    1      1   55 63:63:63:15      yes 3850.0000 1500.0000 1500.0000
 56    1      1   56 36:36:36:9       yes 3850.0000 1500.0000 1500.0000
 57    1      1   57 37:37:37:9       yes 3850.0000 1500.0000 1500.0000
 58    1      1   58 38:38:38:9       yes 3850.0000 1500.0000 1500.0000
 59    1      1   59 39:39:39:9       yes 3850.0000 1500.0000 1500.0000
 60    1      1   60 52:52:52:13      yes 3850.0000 1500.0000 1500.0000
 61    1      1   61 53:53:53:13      yes 3850.0000 1500.0000 1500.0000
 62    1      1   62 54:54:54:13      yes 3850.0000 1500.0000 1500.0000
 63    1      1   63 55:55:55:13      yes 3850.0000 1500.0000 1500.0000
```

## GRUB Configuration
```
$ sonata@coyote2:~$ cat /etc/default/grub
# If you change this file, run 'update-grub' afterwards to update
# /boot/grub/grub.cfg.
# For full documentation of the options in this file, see:
#   info -f grub -n 'Simple configuration'

GRUB_DEFAULT=0
GRUB_TIMEOUT_STYLE=hidden
GRUB_TIMEOUT=0
GRUB_DISTRIBUTOR=`( . /etc/os-release; echo ${NAME:-Ubuntu} ) 2>/dev/null || echo Ubuntu`
GRUB_CMDLINE_LINUX_DEFAULT="amd_iommu=off iommu=pt pci=realloc=off default_hugepagesz=1G hugepagesz=1G hugepages=8 isolcpus=0-3,32-35 nohz_full=0-3,32-35 rcu_nocbs=0-3,32-35 irqaffinity=4-31,36-63 rcu_nocb_poll tsc=reliable audit=0 nosoftlockup"
GRUB_CMDLINE_LINUX=""

# If your computer has multiple operating systems installed, then you
# probably want to run os-prober. However, if your computer is a host
# for guest OSes installed via LVM or raw disk devices, running
# os-prober can cause damage to those guest OSes as it mounts
# filesystems to look for things.
#GRUB_DISABLE_OS_PROBER=false

# Uncomment to enable BadRAM filtering, modify to suit your needs
# This works with Linux (no patch required) and with any kernel that obtains
# the memory map information from GRUB (GNU Mach, kernel of FreeBSD ...)
#GRUB_BADRAM="0x01234567,0xfefefefe,0x89abcdef,0xefefefef"

# Uncomment to disable graphical terminal
#GRUB_TERMINAL=console

# The resolution used on graphical terminal
# note that you can use only modes which your graphic card supports via VBE
# you can see them in real GRUB with the command `vbeinfo'
#GRUB_GFXMODE=640x480

# Uncomment if you don't want GRUB to pass "root=UUID=xxx" parameter to Linux
#GRUB_DISABLE_LINUX_UUID=true

# Uncomment to disable generation of recovery mode menu entries
#GRUB_DISABLE_RECOVERY="true"

# Uncomment to get a beep at grub start
#GRUB_INIT_TUNE="480 440 1"
```

## Persistant Storage
```
$ sonata@coyote2:~$ lsblk
NAME         MAJ:MIN RM   SIZE RO TYPE  MOUNTPOINTS
sda            8:0    1     0B  0 disk
sr0           11:0    1  1024M  0 rom
nvme2n1      259:0    0 931.5G  0 disk
└─md1          9:1    0   7.3T  0 raid0 /mnt/nvme-raid-1
nvme7n1      259:1    0 931.5G  0 disk
└─md1          9:1    0   7.3T  0 raid0 /mnt/nvme-raid-1
nvme11n1     259:2    0 931.5G  0 disk
└─md1          9:1    0   7.3T  0 raid0 /mnt/nvme-raid-1
nvme10n1     259:3    0 931.5G  0 disk
└─md1          9:1    0   7.3T  0 raid0 /mnt/nvme-raid-1
nvme8n1      259:4    0 931.5G  0 disk
└─md1          9:1    0   7.3T  0 raid0 /mnt/nvme-raid-1
nvme5n1      259:5    0 931.5G  0 disk
└─md1          9:1    0   7.3T  0 raid0 /mnt/nvme-raid-1
nvme14n1     259:6    0 931.5G  0 disk
└─md1          9:1    0   7.3T  0 raid0 /mnt/nvme-raid-1
nvme1n1      259:7    0 931.5G  0 disk
└─md0          9:0    0   7.3T  0 raid0 /mnt/nvme-raid-0
nvme6n1      259:8    0 931.5G  0 disk
└─md0          9:0    0   7.3T  0 raid0 /mnt/nvme-raid-0
nvme0n1      259:9    0 931.5G  0 disk
└─md0          9:0    0   7.3T  0 raid0 /mnt/nvme-raid-0
nvme9n1      259:10   0 931.5G  0 disk
└─md0          9:0    0   7.3T  0 raid0 /mnt/nvme-raid-0
nvme4n1      259:11   0 931.5G  0 disk
└─md0          9:0    0   7.3T  0 raid0 /mnt/nvme-raid-0
nvme3n1      259:12   0 931.5G  0 disk
└─md0          9:0    0   7.3T  0 raid0 /mnt/nvme-raid-0
nvme15n1     259:13   0 931.5G  0 disk
└─md0          9:0    0   7.3T  0 raid0 /mnt/nvme-raid-0
nvme16n1     259:14   0 476.9G  0 disk
├─nvme16n1p1 259:17   0     1G  0 part  /boot/efi
└─nvme16n1p2 259:18   0 475.9G  0 part  /
nvme13n1     259:15   0 931.5G  0 disk
└─md1          9:1    0   7.3T  0 raid0 /mnt/nvme-raid-1
nvme12n1     259:16   0 931.5G  0 disk
└─md0          9:0    0   7.3T  0 raid0 /mnt/nvme-raid-0
```

## ConnectX Configuration
```
$ sonata@coyote2:~$ ibv_devinfo
hca_id: mlx5_0
        transport:                      InfiniBand (0)
        fw_ver:                         28.41.1000
        node_guid:                      9c63:c003:00d4:20f0
        sys_image_guid:                 9c63:c003:00d4:20f0
        vendor_id:                      0x02c9
        vendor_part_id:                 4129
        hw_ver:                         0x0
        board_id:                       MT_0000000834
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_DOWN (1)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: mlx5_1
        transport:                      InfiniBand (0)
        fw_ver:                         28.41.1000
        node_guid:                      9c63:c003:00d4:20f1
        sys_image_guid:                 9c63:c003:00d4:20f0
        vendor_id:                      0x02c9
        vendor_part_id:                 4129
        hw_ver:                         0x0
        board_id:                       MT_0000000834
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_DOWN (1)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: mlx5_2
        transport:                      InfiniBand (0)
        fw_ver:                         28.43.2566
        node_guid:                      b8e9:2403:004e:deb0
        sys_image_guid:                 b8e9:2403:004e:deb0
        vendor_id:                      0x02c9
        vendor_part_id:                 4129
        hw_ver:                         0x0
        board_id:                       MT_0000000834
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_DOWN (1)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: mlx5_3
        transport:                      InfiniBand (0)
        fw_ver:                         28.43.2566
        node_guid:                      b8e9:2403:004e:deb1
        sys_image_guid:                 b8e9:2403:004e:deb0
        vendor_id:                      0x02c9
        vendor_part_id:                 4129
        hw_ver:                         0x0
        board_id:                       MT_0000000834
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_DOWN (1)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet
```
## GDS Check
```
$ sonata@coyote2:~$ /usr/local/cuda/gds/tools/gdscheck -p
 GDS release version: 1.14.0.30
 nvidia_fs version:  2.24 libcufile version: 2.12
 Platform: x86_64
 ============
 ENVIRONMENT:
 ============
 =====================
 DRIVER CONFIGURATION:
 =====================
 NVMe P2PDMA        : Unsupported
 NVMe               : Supported
 NVMeOF             : Unsupported
 SCSI               : Unsupported
 ScaleFlux CSD      : Unsupported
 NVMesh             : Unsupported
 DDN EXAScaler      : Unsupported
 IBM Spectrum Scale : Unsupported
 NFS                : Unsupported
 BeeGFS             : Unsupported
 ScaTeFS            : Unsupported
 WekaFS             : Unsupported
 Userspace RDMA     : Unsupported
 --Mellanox PeerDirect : Enabled
 --rdma library        : Not Loaded (libcufile_rdma.so)
 --rdma devices        : Not configured
 --rdma_device_status  : Up: 0 Down: 0
 =====================
 CUFILE CONFIGURATION:
 =====================
 properties.use_pci_p2pdma : false
 properties.use_compat_mode : false
 properties.force_compat_mode : false
 properties.gds_rdma_write_support : true
 properties.use_poll_mode : false
 properties.poll_mode_max_size_kb : 4
 properties.max_batch_io_size : 128
 properties.max_batch_io_timeout_msecs : 5
 properties.max_direct_io_size_kb : 16384
 properties.max_device_cache_size_kb : 131072
 properties.per_buffer_cache_size_kb : 1024
 properties.max_device_pinned_mem_size_kb : 33554432
 properties.posix_pool_slab_size_kb : 4 1024 16384
 properties.posix_pool_slab_count : 128 64 64
 properties.rdma_peer_affinity_policy : RoundRobin
 properties.rdma_dynamic_routing : 0
 fs.generic.posix_unaligned_writes : false
 fs.lustre.posix_gds_min_kb: 0
 fs.beegfs.posix_gds_min_kb: 0
 fs.scatefs.posix_gds_min_kb: 0
 fs.weka.rdma_write_support: false
 fs.gpfs.gds_write_support: false
 fs.gpfs.gds_async_support: true
 profile.nvtx : false
 profile.cufile_stats : 0
 miscellaneous.api_check_aggressive : false
 execution.max_io_threads : 4
 execution.max_io_queue_depth : 128
 execution.parallel_io : true
 execution.min_io_threshold_size_kb : 8192
 execution.max_request_parallelism : 4
 properties.force_odirect_mode : false
 properties.prefer_iouring : false
 =========
 GPU INFO:
 =========
 GPU index 0 NVIDIA RTX 6000 Ada Generation bar:1 bar size (MiB):65536 supports GDS, IOMMU State: Disabled
 GPU index 1 NVIDIA RTX 6000 Ada Generation bar:1 bar size (MiB):65536 supports GDS, IOMMU State: Disabled
 GPU index 2 NVIDIA RTX 6000 Ada Generation bar:1 bar size (MiB):65536 supports GDS, IOMMU State: Disabled
 GPU index 3 NVIDIA RTX 6000 Ada Generation bar:1 bar size (MiB):65536 supports GDS, IOMMU State: Disabled
 ==============
 PLATFORM INFO:
 ==============
 IOMMU: disabled
 Nvidia Driver Info Status: Supported(Nvidia Open Driver Installed)
 Cuda Driver Version Installed:  12090
 Platform: ESC8000A-E12, Arch: x86_64(Linux 6.8.0-64-generic)
 Platform verification succeeded
```
