# Puget Rackstation Threadripper PRO WRX90


## PCIe Topology
```
$ sonata@dev-canyon1:~$ lspci -tv
-+-[0000:00]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14a4
 |           +-00.3  Advanced Micro Devices, Inc. [AMD] Device 14a6
 |           +-01.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-02.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-03.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-03.1-[01-13]----00.0-[02-13]--+-00.0-[03-08]----00.0-[04-08]--+-10.0-[05]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               +-14.0-[06]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               +-18.0-[07]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               \-1c.0-[08]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               +-04.0-[09-0e]----00.0-[0a-0e]--+-00.0-[0b]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               +-04.0-[0c]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               +-08.0-[0d]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               |                               \-0c.0-[0e]----00.0  Samsung Electronics Co Ltd NVMe SSD Controller S4LV008[Pascal]
 |           |                               +-0c.0-[0f-12]----00.0-[10-12]--+-14.0-[11]----00.0  HighPoint Technologies, Inc. SSD7540 PCIe Gen4 x16 8-Port M.2 NVMe RAID Controller
 |           |                               |                               \-15.0-[12]--
 |           |                               \-1c.0-[13]--
 |           +-04.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-05.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-05.1-[14-15]--+-00.0  Intel Corporation Ethernet Controller X710 for 10GBASE-T
 |           |               \-00.1  Intel Corporation Ethernet Controller X710 for 10GBASE-T
 |           +-07.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-07.1-[16]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14ac
 |           |            +-00.4  Advanced Micro Devices, Inc. [AMD] Device 14c9
 |           |            +-00.5  Advanced Micro Devices, Inc. [AMD] Genoa CCP/PSP 4.0 Device
 |           |            \-00.7  Advanced Micro Devices, Inc. [AMD] Device 14cc
 |           +-14.0  Advanced Micro Devices, Inc. [AMD] FCH SMBus Controller
 |           +-14.3  Advanced Micro Devices, Inc. [AMD] FCH LPC Bridge
 |           +-18.0  Advanced Micro Devices, Inc. [AMD] Device 14ad
 |           +-18.1  Advanced Micro Devices, Inc. [AMD] Device 14ae
 |           +-18.2  Advanced Micro Devices, Inc. [AMD] Device 14af
 |           +-18.3  Advanced Micro Devices, Inc. [AMD] Device 14b0
 |           +-18.4  Advanced Micro Devices, Inc. [AMD] Device 14b1
 |           +-18.5  Advanced Micro Devices, Inc. [AMD] Device 14b2
 |           +-18.6  Advanced Micro Devices, Inc. [AMD] Device 14b3
 |           \-18.7  Advanced Micro Devices, Inc. [AMD] Device 14b4
 +-[0000:20]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14a4
 |           +-00.3  Advanced Micro Devices, Inc. [AMD] Device 14a6
 |           +-01.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-01.1-[21]--+-00.0  Mellanox Technologies MT2910 Family [ConnectX-7]
 |           |            \-00.1  Mellanox Technologies MT2910 Family [ConnectX-7]
 |           +-02.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-03.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-03.1-[22-89]----00.0-[23-89]--+-00.0-[28-57]--
 |           |                               +-01.0-[58-87]--
 |           |                               +-02.0-[88]----00.0  ASMedia Technology Inc. Device 2426
 |           |                               \-03.0-[89]----00.0  ASMedia Technology Inc. Device 2425
 |           +-03.2-[8a]----00.0  Kingston Technology Company, Inc. KC3000/FURY Renegade NVMe SSD E18
 |           +-03.4-[8b-8e]----00.0-[8c-8e]--+-0c.0-[8d]----00.0  Advanced Micro Devices, Inc. [AMD] Device 43f8
 |           |                               \-0d.0-[8e]----00.0  Advanced Micro Devices, Inc. [AMD] 600 Series Chipset SATA Controller
 |           +-04.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-05.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-07.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           \-07.1-[8f]----00.0  Advanced Micro Devices, Inc. [AMD] Device 14ac
 +-[0000:c0]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14a4
 |           +-00.3  Advanced Micro Devices, Inc. [AMD] Device 14a6
 |           +-01.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-01.1-[c1]--+-00.0  NVIDIA Corporation AD102GL [RTX 6000 Ada Generation]
 |           |            \-00.1  NVIDIA Corporation AD102 High Definition Audio Controller
 |           +-02.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-03.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-03.1-[c2]--+-00.0  NVIDIA Corporation AD102GL [RTX 6000 Ada Generation]
 |           |            \-00.1  NVIDIA Corporation AD102 High Definition Audio Controller
 |           +-04.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-05.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           +-07.0  Advanced Micro Devices, Inc. [AMD] Device 149f
 |           \-07.1-[c3]----00.0  Advanced Micro Devices, Inc. [AMD] Device 14ac
 \-[0000:e0]-+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14a4
             +-00.3  Advanced Micro Devices, Inc. [AMD] Device 14a6
             +-01.0  Advanced Micro Devices, Inc. [AMD] Device 149f
             +-02.0  Advanced Micro Devices, Inc. [AMD] Device 149f
             +-03.0  Advanced Micro Devices, Inc. [AMD] Device 149f
             +-04.0  Advanced Micro Devices, Inc. [AMD] Device 149f
             +-05.0  Advanced Micro Devices, Inc. [AMD] Device 149f
             +-05.2-[e1-e2]----00.0-[e2]----00.0  ASPEED Technology, Inc. ASPEED Graphics Family
             +-07.0  Advanced Micro Devices, Inc. [AMD] Device 149f
             \-07.1-[e3]--+-00.0  Advanced Micro Devices, Inc. [AMD] Device 14ac
                          \-00.4  Advanced Micro Devices, Inc. [AMD] Device 14c9
```

## NUMA Topology
```
$ sonata@dev-canyon1:~$ numactl --hardware
available: 1 nodes (0)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
node 0 size: 128295 MB
node 0 free: 1724 MB
node distances:
node   0
  0:  10
```

## PCIe Devices Details
GPU #0
```
$ sonata@dev-canyon1:~$ sudo lspci -vvv -s c1:00.0
[sudo] password for sonata:
c1:00.0 VGA compatible controller: NVIDIA Corporation AD102GL [RTX 6000 Ada Generation] (rev a1) (prog-if 00 [VGA controller])
        Subsystem: NVIDIA Corporation AD102GL [RTX 6000 Ada Generation]
        Control: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0
        Interrupt: pin A routed to IRQ 359
        Region 0: Memory at b4000000 (32-bit, non-prefetchable) [size=16M]
        Region 1: Memory at 13000000000 (64-bit, prefetchable) [size=64G]
        Region 3: Memory at 14000000000 (64-bit, prefetchable) [size=32M]
        Region 5: I/O ports at 5000 [size=128]
        Expansion ROM at b5000000 [virtual] [disabled] [size=512K]
        Capabilities: [60] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=0mA PME(D0+,D1-,D2-,D3hot+,D3cold-)
                Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [68] MSI: Enable+ Count=1/1 Maskable- 64bit+
                Address: 00000000fee10000  Data: 002b
        Capabilities: [78] Express (v2) Legacy Endpoint, MSI 00
                DevCap: MaxPayload 256 bytes, PhantFunc 0, Latency L0s unlimited, L1 <64us
                        ExtTag+ AttnBtn- AttnInd- PwrInd- RBE+ FLReset+
                DevCtl: CorrErr+ NonFatalErr+ FatalErr+ UnsupReq+
                        RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop+ FLReset-
                        MaxPayload 256 bytes, MaxReadReq 512 bytes
                DevSta: CorrErr- NonFatalErr- FatalErr- UnsupReq- AuxPwr- TransPend-
                LnkCap: Port #0, Speed 16GT/s, Width x16, ASPM L1, Exit Latency L1 <4us
                        ClockPM+ Surprise- LLActRep- BwNot- ASPMOptComp+
                LnkCtl: ASPM Disabled; RCB 64 bytes, Disabled- CommClk+
                        ExtSynch- ClockPM+ AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 2.5GT/s (downgraded), Width x16
                        TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-
                DevCap2: Completion Timeout: Range AB, TimeoutDis+ NROPrPrP- LTR-
                         10BitTagComp+ 10BitTagReq+ OBFF Via message, ExtFmt- EETLPPrefix-
                         EmergencyPowerReduction Not Supported, EmergencyPowerReductionInit-
                         FRS-
                         AtomicOpsCap: 32bit- 64bit- 128bitCAS-
                DevCtl2: Completion Timeout: 65ms to 210ms, TimeoutDis- LTR- 10BitTagReq+ OBFF Via message B,
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
                        Ctrl:   Enable+ ID=0 ArbSelect=Fixed TC/VC=ff
                        Status: NegoPending- InProgress-
        Capabilities: [258 v1] L1 PM Substates
                L1SubCap: PCI-PM_L1.2+ PCI-PM_L1.1+ ASPM_L1.2- ASPM_L1.1+ L1_PM_Substates+
                          PortCommonModeRestoreTime=255us PortTPowerOnTime=10us
                L1SubCtl1: PCI-PM_L1.2- PCI-PM_L1.1- ASPM_L1.2- ASPM_L1.1-
                           T_CommonMode=0us
                L1SubCtl2: T_PwrOn=150us
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
$ sonata@dev-canyon1:~$ sudo lspci -vvv -s c2:00.0
c2:00.0 VGA compatible controller: NVIDIA Corporation AD102GL [RTX 6000 Ada Generation] (rev a1) (prog-if 00 [VGA controller])
        Subsystem: NVIDIA Corporation AD102GL [RTX 6000 Ada Generation]
        Control: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0
        Interrupt: pin A routed to IRQ 360
        Region 0: Memory at b2000000 (32-bit, non-prefetchable) [size=16M]
        Region 1: Memory at 11000000000 (64-bit, prefetchable) [size=64G]
        Region 3: Memory at 12000000000 (64-bit, prefetchable) [size=32M]
        Region 5: I/O ports at 4000 [size=128]
        Expansion ROM at b3000000 [virtual] [disabled] [size=512K]
        Capabilities: [60] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=0mA PME(D0+,D1-,D2-,D3hot+,D3cold-)
                Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [68] MSI: Enable+ Count=1/1 Maskable- 64bit+
                Address: 00000000fee12000  Data: 002c
        Capabilities: [78] Express (v2) Legacy Endpoint, MSI 00
                DevCap: MaxPayload 256 bytes, PhantFunc 0, Latency L0s unlimited, L1 <64us
                        ExtTag+ AttnBtn- AttnInd- PwrInd- RBE+ FLReset+
                DevCtl: CorrErr+ NonFatalErr+ FatalErr+ UnsupReq+
                        RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop+ FLReset-
                        MaxPayload 256 bytes, MaxReadReq 512 bytes
                DevSta: CorrErr- NonFatalErr- FatalErr- UnsupReq- AuxPwr- TransPend-
                LnkCap: Port #0, Speed 16GT/s, Width x16, ASPM L1, Exit Latency L1 <4us
                        ClockPM+ Surprise- LLActRep- BwNot- ASPMOptComp+
                LnkCtl: ASPM Disabled; RCB 64 bytes, Disabled- CommClk+
                        ExtSynch- ClockPM+ AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 2.5GT/s (downgraded), Width x16
                        TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-
                DevCap2: Completion Timeout: Range AB, TimeoutDis+ NROPrPrP- LTR-
                         10BitTagComp+ 10BitTagReq+ OBFF Via message, ExtFmt- EETLPPrefix-
                         EmergencyPowerReduction Not Supported, EmergencyPowerReductionInit-
                         FRS-
                         AtomicOpsCap: 32bit- 64bit- 128bitCAS-
                DevCtl2: Completion Timeout: 65ms to 210ms, TimeoutDis- LTR- 10BitTagReq+ OBFF Via message B,
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
                        Ctrl:   Enable+ ID=0 ArbSelect=Fixed TC/VC=ff
                        Status: NegoPending- InProgress-
        Capabilities: [258 v1] L1 PM Substates
                L1SubCap: PCI-PM_L1.2+ PCI-PM_L1.1+ ASPM_L1.2- ASPM_L1.1+ L1_PM_Substates+
                          PortCommonModeRestoreTime=255us PortTPowerOnTime=10us
                L1SubCtl1: PCI-PM_L1.2- PCI-PM_L1.1- ASPM_L1.2- ASPM_L1.1-
                           T_CommonMode=0us
                L1SubCtl2: T_PwrOn=150us
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
ConnectX-7
```
$ sonata@dev-canyon1:~$ sudo lspci -vvv -s 21:00.0
21:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
        Subsystem: Mellanox Technologies MT2910 Family [ConnectX-7]
        Control: I/O- Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0, Cache Line Size: 64 bytes
        Interrupt: pin A routed to IRQ 27
        Region 0: Memory at 18802000000 (64-bit, prefetchable) [size=32M]
        Expansion ROM at b0c00000 [disabled] [size=1M]
        Capabilities: [60] Express (v2) Endpoint, MSI 00
                DevCap: MaxPayload 512 bytes, PhantFunc 0, Latency L0s unlimited, L1 unlimited
                        ExtTag+ AttnBtn- AttnInd- PwrInd- RBE+ FLReset+ SlotPowerLimit 75W
                DevCtl: CorrErr+ NonFatalErr+ FatalErr+ UnsupReq+
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
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis- LTR- 10BitTagReq+ OBFF Disabled,
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
                        [SN] Serial number: MT2421600CDE
                        [V3] Vendor specific: 2c60caffaa17ef1180009c63c0d5edf6
                        [VA] Vendor specific: MLX:MN=MLNX:CSKU=V2:UUID=V3:PCI=V0:MODL=CX755106A
                        [V0] Vendor specific: PCIeGen5 x16
                        [VU] Vendor specific: MT2421600CDEMLNXS0D0F0
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
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq+ ACSViol-
                UESvrt: DLP+ SDES- TLP- FCP+ CmpltTO+ CmpltAbrt- UnxCmplt+ RxOF+ MalfTLP+ ECRC+ UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover+ Timeout+ AdvNonFatalErr-
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
                Region 0: Memory at 0000018806000000 (64-bit, prefetchable)
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
HighPoint Carrier Board
```
$ sonata@dev-canyon1:~$ sudo lspci -vvv -s 11:00.0
11:00.0 RAID bus controller: HighPoint Technologies, Inc. SSD7540 PCIe Gen4 x16 8-Port M.2 NVMe RAID Controller (rev 01)
        Subsystem: HighPoint Technologies, Inc. SSD7540 PCIe Gen4 x16 8-Port M.2 NVMe RAID Controller
        Control: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx-
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0, Cache Line Size: 64 bytes
        Interrupt: pin A routed to IRQ 255
        Region 0: I/O ports at 1050 [size=8]
        Region 1: I/O ports at 1040 [size=4]
        Region 2: I/O ports at 1030 [size=8]
        Region 3: I/O ports at 1020 [size=4]
        Region 4: I/O ports at 1000 [size=32]
        Region 5: Memory at f7810000 (32-bit, non-prefetchable) [size=2K]
        Expansion ROM at f7800000 [disabled] [size=64K]
        Capabilities: [40] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=0mA PME(D0-,D1-,D2-,D3hot+,D3cold-)
                Status: D0 NoSoftRst- PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [50] MSI: Enable- Count=1/1 Maskable- 64bit-
                Address: 00000000  Data: 0000
        Capabilities: [70] Express (v2) Legacy Endpoint, MSI 00
                DevCap: MaxPayload 512 bytes, PhantFunc 0, Latency L0s <1us, L1 <8us
                        ExtTag- AttnBtn- AttnInd- PwrInd- RBE+ FLReset-
                DevCtl: CorrErr+ NonFatalErr+ FatalErr+ UnsupReq+
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
$ sonata@dev-canyon1:~/stelline$ nvidia-smi
Tue May 27 21:19:19 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.51.03              Driver Version: 575.51.03      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:C1:00.0 Off |                  Off |
| 30%   43C    P8              8W /  300W |       4MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:C2:00.0 Off |                  Off |
| 30%   43C    P8             13W /  300W |       4MiB /  49140MiB |      0%      Default |
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
$ sonata@dev-canyon1:~/stelline$ nvidia-smi topo -m
        GPU0    GPU1    NIC0    NIC1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      PHB     NODE    NODE    0-23    0               N/A
GPU1    PHB      X      NODE    NODE    0-23    0               N/A
NIC0    NODE    NODE     X      PIX
NIC1    NODE    NODE    PIX      X

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
```

## NVIDIA Transfers
```
$ sonata@dev-canyon1:/1_Utilities/bandwidthTest$ ./bandwidthTest
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
   32000000                     4374.1

Result = PASS
```

```
$ sonata@dev-canyon1:/5_Domain_Specific/p2pBandwidthLatencyTest$ ./p2pBandwidthLatencyTest
[P2P (Peer-to-Peer) GPU Bandwidth Latency Test]
Device: 0, NVIDIA RTX 6000 Ada Generation, pciBusID: c1, pciDeviceID: 0, pciDomainID:0
Device: 1, NVIDIA RTX 6000 Ada Generation, pciBusID: c2, pciDeviceID: 0, pciDomainID:0
Device=0 CAN Access Peer Device=1
Device=1 CAN Access Peer Device=0

***NOTE: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.
So you can see lesser Bandwidth (GB/s) and unstable Latency (us) in those cases.

P2P Connectivity Matrix
     D\D     0     1
     0       1     1
     1       1     1
Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 801.28  22.23
     1  22.32 813.80
Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   D\D     0      1
     0 800.15  26.32
     1  26.32 831.12
Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 800.29  31.42
     1  31.45 808.33
Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 802.94  52.11
     1  52.10 806.22
P2P=Disabled Latency Matrix (us)
   GPU     0      1
     0   1.39  10.90
     1  10.46   1.45

   CPU     0      1
     0   1.62   4.74
     1   4.86   1.51
P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1
     0   1.39   1.12
     1   1.14   1.42

   CPU     0      1
     0   1.64   1.32
     1   1.32   1.58
```

## RDMA P2P Transfers
```
$ sonata@dev-canyon1:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid/test.bin -x 0 -d 0 -s 50G -I 1 -i 1G
IoType: WRITE XferType: GPUD Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 25.112706 GiB/sec, Avg_Latency: 38738.000000 usecs ops: 50 total_time 1.991024 secs
$ sonata@dev-canyon1:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid/test.bin -x 0 -d 1 -s 50G -I 1 -i 1G
IoType: WRITE XferType: GPUD Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 25.034021 GiB/sec, Avg_Latency: 38861.360000 usecs ops: 50 total_time 1.997282 secs
$ sonata@dev-canyon1:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid/test.bin -x 1 -d 0 -s 50G -I 1 -i 1G
IoType: WRITE XferType: CPUONLY Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 25.775109 GiB/sec, Avg_Latency: 37840.940000 usecs ops: 50 total_time 1.939856 secs
$ sonata@dev-canyon1:~$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid/test.bin -x 1 -d 1 -s 50G -I 1 -i 1G
IoType: WRITE XferType: CPUONLY Threads: 1 DataSetSize: 52428800/52428800(KiB) IOSize: 1048576(KiB) Throughput: 25.797158 GiB/sec, Avg_Latency: 37814.600000 usecs ops: 50 total_time 1.938198 secs
```

## ACS State
```
$ sonata@dev-canyon1:~$ sudo lspci -vvv | grep ACSCtl
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
$ sonata@dev-canyon1:~$ lscpu
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          52 bits physical, 57 bits virtual
  Byte Order:             Little Endian
CPU(s):                   24
  On-line CPU(s) list:    0-23
Vendor ID:                AuthenticAMD
  Model name:             AMD Ryzen Threadripper PRO 7965WX 24-Cores
    CPU family:           25
    Model:                24
    Thread(s) per core:   1
    Core(s) per socket:   24
    Socket(s):            1
    Stepping:             1
    CPU(s) scaling MHz:   19%
    CPU max MHz:          5362.0000
    CPU min MHz:          545.0000
    BogoMIPS:             8386.98
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd
                          _apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs
                          skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rd
                          t_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clze
                          ro irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc amd_ibpb_ret arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic vgif x2avic v_spec_ctrl vnmi avx51
                          2vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d debug_swap
Virtualization features:
  Virtualization:         AMD-V
Caches (sum of all):
  L1d:                    768 KiB (24 instances)
  L1i:                    768 KiB (24 instances)
  L2:                     24 MiB (24 instances)
  L3:                     128 MiB (4 instances)
NUMA:
  NUMA node(s):           1
  NUMA node0 CPU(s):      0-23
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
$ sonata@dev-canyon1:~$ lscpu -e
CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ       MHZ
  0    0      0    0 0:0:0:0          yes 5362.0000 545.0000  545.0000
  1    0      0    1 1:1:1:0          yes 5362.0000 545.0000  545.0000
  2    0      0    2 2:2:2:0          yes 5362.0000 545.0000  545.0000
  3    0      0    3 3:3:3:0          yes 5362.0000 545.0000  545.0000
  4    0      0    4 4:4:4:0          yes 5362.0000 545.0000 4783.6440
  5    0      0    5 5:5:5:0          yes 5362.0000 545.0000  545.0000
  6    0      0    6 16:16:16:2       yes 5362.0000 545.0000  545.0000
  7    0      0    7 17:17:17:2       yes 5362.0000 545.0000  545.0000
  8    0      0    8 18:18:18:2       yes 5362.0000 545.0000  545.0000
  9    0      0    9 19:19:19:2       yes 5362.0000 545.0000  545.0000
 10    0      0   10 20:20:20:2       yes 5362.0000 545.0000  545.0000
 11    0      0   11 21:21:21:2       yes 5362.0000 545.0000  545.0000
 12    0      0   12 32:32:32:4       yes 5362.0000 545.0000  545.0000
 13    0      0   13 33:33:33:4       yes 5362.0000 545.0000  545.0000
 14    0      0   14 34:34:34:4       yes 5362.0000 545.0000  545.0000
 15    0      0   15 35:35:35:4       yes 5362.0000 545.0000 1793.5580
 16    0      0   16 36:36:36:4       yes 5362.0000 545.0000  545.0000
 17    0      0   17 37:37:37:4       yes 5362.0000 545.0000  545.0000
 18    0      0   18 48:48:48:6       yes 5362.0000 545.0000 4792.0000
 19    0      0   19 49:49:49:6       yes 5362.0000 545.0000  545.0000
 20    0      0   20 50:50:50:6       yes 5362.0000 545.0000  545.0000
 21    0      0   21 51:51:51:6       yes 5362.0000 545.0000  545.0000
 22    0      0   22 52:52:52:6       yes 5362.0000 545.0000 2930.3811
 23    0      0   23 53:53:53:6       yes 5362.0000 545.0000  545.0000
```

## GRUB Configuration
```
$ sonata@dev-canyon1:~$ cat /etc/default/grub
# If you change this file, run 'update-grub' afterwards to update
# /boot/grub/grub.cfg.
# For full documentation of the options in this file, see:
#   info -f grub -n 'Simple configuration'

GRUB_DEFAULT=0
GRUB_TIMEOUT_STYLE=hidden
GRUB_TIMEOUT=0
GRUB_DISTRIBUTOR=`( . /etc/os-release; echo ${NAME:-Ubuntu} ) 2>/dev/null || echo Ubuntu`
GRUB_CMDLINE_LINUX_DEFAULT="amd_iommu=off iommu=pt pci=realloc=off isolcpus=0-3 nohz_full=0-3 rcu_nocbs=0-3 irqaffinity=4-23 rcu_nocb_poll tsc=reliable audit=0 nosoftlockup default_hugepagesz=1G hugepagesz=1G hugepages=8"
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
$ sonata@dev-canyon1:~$ lsblk
NAME                      MAJ:MIN RM   SIZE RO TYPE  MOUNTPOINTS
nvme0n1                   259:0    0 931.5G  0 disk
└─md127                     9:127  0   7.3T  0 raid0 /mnt/nvme-raid
nvme3n1                   259:1    0 931.5G  0 disk
└─md127                     9:127  0   7.3T  0 raid0 /mnt/nvme-raid
nvme6n1                   259:2    0   1.9T  0 disk
├─nvme6n1p1               259:9    0     1G  0 part  /boot/efi
├─nvme6n1p2               259:10   0     2G  0 part  /boot
└─nvme6n1p3               259:11   0   1.9T  0 part
  └─ubuntu--vg-ubuntu--lv 252:0    0   1.9T  0 lvm   /
nvme7n1                   259:3    0 931.5G  0 disk
└─md127                     9:127  0   7.3T  0 raid0 /mnt/nvme-raid
nvme4n1                   259:4    0 931.5G  0 disk
└─md127                     9:127  0   7.3T  0 raid0 /mnt/nvme-raid
nvme2n1                   259:5    0 931.5G  0 disk
└─md127                     9:127  0   7.3T  0 raid0 /mnt/nvme-raid
nvme5n1                   259:6    0 931.5G  0 disk
└─md127                     9:127  0   7.3T  0 raid0 /mnt/nvme-raid
nvme1n1                   259:7    0 931.5G  0 disk
└─md127                     9:127  0   7.3T  0 raid0 /mnt/nvme-raid
nvme8n1                   259:8    0 931.5G  0 disk
└─md127                     9:127  0   7.3T  0 raid0 /mnt/nvme-raid
```

## ConnectX Configuration
```
$ sonata@dev-canyon1:~$ ibv_devinfo
hca_id: mlx5_0
        transport:                      InfiniBand (0)
        fw_ver:                         28.44.1036
        node_guid:                      9c63:c003:00d5:edf6
        sys_image_guid:                 9c63:c003:00d5:edf6
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
        fw_ver:                         28.44.1036
        node_guid:                      9c63:c003:00d5:edf7
        sys_image_guid:                 9c63:c003:00d5:edf6
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
```
## GDS Check
```
$ sonata@dev-canyon1:~$ /usr/local/cuda/gds/tools/gdscheck -p
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
 ==============
 PLATFORM INFO:
 ==============
 IOMMU: disabled
 Nvidia Driver Info Status: Supported(Nvidia Open Driver Installed)
 Cuda Driver Version Installed:  12090
 Platform: Puget Rackstation Threadripper PRO WRX90 T141-4U, Arch: x86_64(Linux 6.8.0-60-generic)
 Platform verification succeeded
```
