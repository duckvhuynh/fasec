# Face liveness landscape: what Windows Hello, Face ID and Android do, and what fasec should do

Status: research note, 2026-11-22. No code. Inputs for the IR liveness plan
(`docs/superpowers/plans/2026-11-22-ir-liveness.md`) and PRD LV-3 / LV-5.

Labels: **measured** = run on the dev laptop for this note. **inferred** = reasoning from cited
facts, not tested. Everything else carries a source URL.

## Summary

1. **Windows Hello face is 2D NIR, not depth.** It needs an IR stream of at least 320x320 with
   alternating lit/unlit frames (or ambient subtraction) at 15+15 fps. Since April 2025 it also
   requires the color camera to see a face. fasec's IR tier can follow the same recipe on the same
   hardware. [MS-HELLO-DRV], [CVE-2025-26644]
2. **The "screens are black in NIR" cue is necessary but no longer enough.** Red Bleed (USENIX
   Security 2025) beat Windows Hello with an LCD rebuilt to emit 850 nm. Microsoft's fix was
   RGB+IR fusion. fasec must fuse too: IR evidence alone never yields `Live`. [RED-BLEED]
3. **Lit-minus-unlit differencing is the core IR cue.** A real face reflects the emitter. A
   NIR-emitting screen or a lamp-lit print shows up in the unlit frame as well (inferred). This
   only works if the driver delivers real unlit frames. On the dev laptop the unlit frames have
   mean 0.0, so that has to be checked first (open question 2).
4. **ESS is ON on the dev laptop, and a normal app still reads IR frames** (measured). Windows
   Hello's face sensor runs in a Virtual Secure Mode process and both cameras report the
   secure-device capability. SP-7 still read IR frames with `SharedReadOnly`, without admin.
   ESS protects the Hello pipeline. It does not hide the camera from other apps the rest of the
   time. Untested: frames during an active Hello sign-in.
5. **Depth is not needed now.** Windows Hello (2D NIR + RGB) and Pixel 8 (RGB only, Class 3)
   reach their security bar without depth. Consumer Windows laptops with depth sensors are
   essentially gone. Keep depth as an optional P2 plugin (`MediaFrameSourceKind::Depth`, RealSense
   or Orbbec, both permissive SDKs). Do not design around it. [PIXEL8], [MS-FRAMES]
6. **RGB-only liveness should be labelled as a deterrent, not anti-spoofing.** NIST tested 82
   passive 2D PAD algorithms and found none that caught every attack type. Commercial RGB vendors
   add active illumination (iProov Flashmark, Azure Passive-Active), server-side models and
   injection defenses that fasec does not have. [NIST-8491], [AZURE-LV]
7. **State claims in ISO/IEC 30107-3 terms: APCER per attack species, plus BPCER.** Never one
   "accuracy" number. Report the tier, the species tested (print, phone replay, monitor replay,
   NIR print) and N. Do not use "iBeta Level 1" or "Class 2" unless a lab tested it.
   [ISO-30107-3], [IBETA]
8. **Reusable assets.** The current stack is already permissive: YuNet (MIT), SFace
   (Apache-2.0), MiniFASNet (Apache-2.0). There is no permissive, pretrained NIR or RGB+IR
   anti-spoof model worth adopting. Every RGB+IR model is trained on CASIA-SURF, WMCA or CeFA,
   which are research-only datasets. Use hand-built cues first and our own data later.
9. **Recommended tiers.**
   - **`ir`** (prefer when the paired IR source exists and is healthy; setting
     `liveness.prefer_ir = true` by default): IR face present in the lit frame, absent or much
     weaker in the unlit frame, NIR shading consistent with a 3D surface, IR box geometrically
     consistent with the RGB box, plus temporal cues. An IR spoof verdict overrides RGB `Live`.
   - **`rgb_temporal` / `rgb`**: MiniFASNet plus blink and motion. Labelled honestly in the UI.
   - **`depth`**: optional plugin later. Never a prerequisite.
10. **Injection is out of scope, but name it.** A custom USB device can feed Windows Hello
    a stored IR frame (CyberArk, CVE-2021-34466). fasec cannot defend against hostile hardware
    and should say so. It can prefer an RGB and IR source on the same physical USB device.
    [CYBERARK]

## 1. Windows Hello face

**Hardware.** Hello face uses "a camera specially configured for near infrared (IR) imaging"
[MS-HELLO-FACE]. HLK tests require exactly one FaceAuth-capable stream, IR only, L8/NV12/MJPG,
at least 320x320. They require one of two FaceAuth modes: alternating illuminated/un-illuminated
frames (`KSCAMERA_EXTENDEDPROP_FACEAUTH_MODE_ALTERNATIVE_FRAME_ILLUMINATION`) or ambient
subtraction. The rate is 15 fps lit + 15 fps ambient, or 15 fps ambient-subtracted. Startup must
be under 500 ms. Lit frames carry `MF_CAPTURE_METADATA_FRAME_ILLUMINATION` [MS-HELLO-DRV]. Apps
see this as `InfraredMediaFrame.IsIlluminated` [MS-ISILLUM]. Nothing in the requirements
mentions depth. Hello is a 2D NIR system. Early Hello cameras such as the Intel RealSense F200
were depth cameras, but the requirements never needed depth [RS-F200].

**Matching and anti-spoofing (public level only).** The pipeline is landmarks, a head pose
check of ±15°, a representation vector and a learned threshold. The stated FAR is under 1/100,000
[MS-HELLO-FACE]. On spoofing, Microsoft says only that IR "helps prevent the most accessible
attacks", because photos and LCDs do not show the IR appearance [MS-HELLO-FACE]. Admins can turn
on "enhanced anti-spoofing", which is stricter and fails more in dark rooms [MS-HELLO-SUPPORT].

**History of breaks.** Each one changed the design, so they are lessons for fasec:
- 2021, CyberArk: only IR frames were processed. A custom USB "camera" that sent one IR frame
  of the victim plus a black RGB frame unlocked the PC. Fixed in CVE-2021-34466 [CYBERARK],
  [KB5005478].
- 2025, Red Bleed (NTU): an LCD with an 850 nm backlight and polarizer replayed an IR video of the
  victim and beat Hello [RED-BLEED]. MSRC assigned CVE-2025-26644. Since April 2025,
  "facial recognition requires color cameras to see a visible face". Face unlock in the dark broke
  as a result [CVE-2025-26644], [TH-DARK].
- Lesson (inferred): Microsoft ended up with RGB+IR consistency. fasec should start there.

**Enhanced Sign-in Security (ESS).** With ESS, "the face algorithm is protected using VBS". The
hypervisor lets "the face camera write to these memory regions providing an isolated pathway" to
the matcher. Templates are encrypted with VBS-only keys [MS-ESS]. Requirements: VBS + TPM 2.0,
an ESS-capable IR camera (firmware support, specific chipsets, the inbox UVC driver) and an SDEV
ACPI table. Device Manager shows `CM_DEVCAP_SECUREDEVICE` on the camera and the xHCI controller
[MS-ESS]. **All Copilot+ PCs ship with ESS on.** Other PCs enable it at the factory or when the
first sensor is enrolled [MS-ESS]. External cameras are not supported for ESS sign-in, but
"you can still use external peripherals within apps like Teams" [MS-ESS].

**Can a third-party app read IR on an ESS machine?** Microsoft does not document this. The
measured answer on the dev laptop, an ASUS TUF Gaming F16 FX608JMR on Windows 11 build 26300:
- VBS running (`Win32_DeviceGuard.VirtualizationBasedSecurityStatus = 2`). `NgcIso` is running.
- Biometrics/Operational event 1108: "Facial Recognition (Windows Hello) Software Device ...
  isolated in a 'Virtual Secure Mode' process". Microsoft names this as the ESS-on signal
  [MS-ESS].
- `ASUS IR camera` and `ASUS FHD webcam` report `DEVPKEY_Device_Capabilities = 0x4A4`, which
  includes `CM_DEVCAP_SECUREDEVICE (0x400)` (value checked in the local SDK `cfgmgr32.h`). The
  USB 3.20 xHCI controller reports `0x480`.
- SP-7 (`docs/spikes/SP-7-ir.md`) read 30 IR frames with `SharedReadOnly`, without admin, from
  both the IR-only group and the RGB+IR group.
- Conclusion (measured + inferred): ESS isolates frames on Hello's own capture path. It does
  not stop shared-read access by other apps while Hello is idle. Unknown: what fasec gets during
  an active Hello sign-in or "Improve recognition". It may get stalls or no frames, so fasec
  must treat an IR stall as `Lost` and fall back. This matches Review Focus 3 of the IR plan.

## 2. Apple Face ID

The TrueDepth system "projects and reads thousands of infrared dots to form a depth map of the
face along with a 2D infrared image". It randomizes the order of 2D and depth captures and
projects "a device-specific random pattern" [APPLE-SEC]. Neural networks handle "attention,
matching, and antispoofing" inside the Secure Enclave. Attention means eyes open and looking at
the device. It can be switched off, and it is always on with a mask [APPLE-SEC]. False match:
"less than 1 in 1,000,000". It is weaker for twins, siblings and children under 13 [APPLE-FACEID].

**What depth buys over 2D NIR.** Depth makes print and screen attacks fail on geometry. Paper
bent around a head still fails the fine structure of nose, eye sockets and cheeks. The random
per-device pattern defeats replay of captured sensor output. Masks remain the residual threat.
Apple describes anti-spoofing networks built against masks [APPLE-FACEID]. The cost is a dot projector, a flood illuminator,
calibration and a secure pipeline. No Windows laptop has this (§8). 2D NIR with RGB fusion covers
the print and screen attacks that matter for fasec. Depth adds mask resistance that a privacy
guard does not need (inferred).

## 3. Android biometric classes

Android grades biometrics by spoof acceptance rate (SAR) [AOSP-MEASURE]:

| Class | SAR, all PAI species | Level A / Level B | FAR / FRR | Effect |
|---|---|---|---|---|
| Class 3 (Strong) | 0-7% | ≤7% / ≤20% | 1/50k / 10% | Keystore and BiometricPrompt, 72 h before fallback |
| Class 2 (Weak) | 7-20% | ≤20% / ≤30% | 1/50k / 10% | BiometricPrompt, no Keystore, 24 h |
| Class 1 (Convenience) | 20-30% | ≤30% / ≤40% | 1/50k / 10% | Unlock only, no app APIs |

Face PAI species in scope are 2D (print, screen photo, video) and 3D-printed masks. Silicone
masks are out of scope. The test uses at least 10 subjects with at least 10 attempts each
[AOSP-MEASURE]. 2D-only phones used to be Class 1. Pixel 8 reached Class 3 with the ordinary
selfie camera plus on-device ML on Tensor G3, without depth hardware. Pixel 4 had needed dedicated face
hardware for the same class [PIXEL8]. Lesson: the bar is measured SAR per species, not sensor type.
fasec should copy the method (per-species attack sets, ≥10×10) even if it never claims a class.

## 4. ISO/IEC 30107-3 metrics and honest claims

- **APCER**: the share of attack presentations accepted as bona fide, computed **per PAI
  species**. The reported value is the worst species. **BPCER**: the share of bona fide
  presentations rejected as attacks. Lowering one raises the other [ISO-30107-3],
  [PAD-METRICS].
- Attack potential levels are lab conventions built on 30107-3, not part of the standard.
  iBeta Level 1: about 8 h per species, home or office materials, artefacts under $30, 0%
  penetration allowed. Level 2: 2-4 days, artefacts under $300 (3D prints, resin or latex masks),
  1% allowed. BPCER must be ≤15% at both levels [IBETA]. FIDO uses IAPAR, the impostor attack
  accept rate, with labs testing per ISO/IEC 30107-4 [FIDO-BIO].
- **How fasec should word it.** Example: "IR tier, dev laptop X, 10 subjects: APCER print 0/100,
  phone replay 0/100, monitor replay 0/100, NIR print k/100. BPCER m/200 at threshold t. Not
  evaluated by an accredited lab." Never write "certified", "iBeta" or "Class 3" without a
  lab report.

## 5. IR/NIR PAD techniques for one 2D NIR camera plus RGB

| Technique | Hardware | vs print | vs screen replay | vs mask | Cost in fasec |
|---|---|---|---|---|---|
| IR face presence (RGB face, no IR face → spoof) | IR + RGB | weak (paper reflects NIR) | strong vs normal LCD/OLED, fails vs NIR screen [RED-BLEED] | none | Low. YuNet on lit frame plus association (plan Task 2) |
| Lit − unlit differencing (face must appear only when emitter is on) | IR with alternating frames | medium (lamp/sun-lit print shows in unlit) | strong vs self-emitting NIR screen (inferred) | weak | Low, **if** unlit frames are real (open q. 2) |
| NIR reflectance/texture of skin vs paper, ink, plastic | IR | medium-strong [NIR-REFL], [NIR-SKIN] | strong | medium (material dependent) | Medium. Hand features first, small model later |
| Shading from a point emitter (inverse-square falloff, nose highlight, cheek gradient) | IR | strong vs flat print (inferred) | strong | weak | Medium. Ratio features over landmarks |
| Bright pupil / corneal glint from the near-axis emitter | IR, emitter close to lens | strong vs print (no retro-reflection) | strong | medium | Medium. Strength varies by person and glasses [BRIGHT-PUPIL], [IBM-PUPIL] |
| RGB–NIR consistency (same landmarks, pose, blink; plausible cross-modal appearance) | IR + RGB, same device | strong | strong, including NIR screen (needs two displays in sync) | medium | Low-medium. Affine map plus a per-track agreement score |
| Temporal: blink in IR, micro-motion, landmark jitter vs rigid planar motion | IR or RGB | medium | weak vs video | weak | Low. Already partly in the RGB tier |
| RGB MiniFASNet | RGB | medium on trained data | medium | weak | Done. Fails at close range (plan "Why now") |
| Screen-flash / color-challenge reflections (iProov, Azure Active) | RGB + display | strong | strong | medium | High, and visibly intrusive for a background guard. Skip |
| Depth geometry | depth camera | strong | strong | medium | High. Plugin only (§8) |
| SWIR / thermal | special sensors | strong | strong | strong [HQ-WMCA-SWIR] | Not available on laptops. Never |

Notes:
- Microsoft's FaceAuth modes (alternating illumination or ambient subtraction) are differencing
  at the driver level [MS-HELLO-DRV]. With ambient subtraction the app only gets the difference,
  and the unlit-frame cue is lost.
- Paper reflects 850 nm well. A plain RGB print looks "washed out" in NIR rather than missing
  [NIR-REFL]. An attacker who prints an NIR capture of the victim is the realistic print attack,
  as in CyberArk's IR frame [CYBERARK]. Shading, glint and RGB consistency are what catch it.
- Datasets, all **research-only**: CASIA-SURF (1,000 subjects, 21k videos, RGB+depth+IR, print
  and cut attacks) [CASIA-SURF]; CASIA-SURF CeFA (signed agreement, commercial license sold
  separately) [CEFA-DL]; WMCA (72 identities, 1,941 videos, color/depth/IR/thermal, non-commercial
  EULA) [WMCA], [WMCA-ZENODO]; HQ-WMCA (adds SWIR) [HQ-WMCA]. None can train a model fasec ships
  under a permissive license. They are fine for reading and for offline benchmarking.

## 6. Reusable open-source assets

| Asset | What | License | Verdict |
|---|---|---|---|
| YuNet (opencv_zoo / libfacedetection) | Face detector, 5 landmarks | MIT (model dir) [YUNET-LIC] | **Use** (already). NIR accuracy unverified, see open q. 3 |
| SFace (opencv_zoo) | Face embedding | Apache-2.0 [SFACE-LIC] | **Use** (already) |
| Silent-Face-Anti-Spoofing (MiniFASNet) | RGB passive PAD, weights in repo | Apache-2.0 [GH-SILENT] | **Use** (already). Training data undisclosed |
| yakhyo/face-anti-spoofing, UniFace | MiniFASNet inference and exports | Apache-2.0 / MIT [GH-YAKHYO] | Optional reference for ONNX export |
| MediaPipe Face Detector / Landmarker | BlazeFace and 478-point mesh | Apache-2.0 [GH-MEDIAPIPE] | Use if eye and iris landmarks are needed for glint and blink |
| kprokofi/light-weight-face-anti-spoofing | RGB PAD (MobileNetV3) | MIT code [GH-KPROKOFI] | **Caution**: weights trained on CelebA-Spoof (non-commercial). Code only |
| AlexanderParkin/ChaLearn_liveness_challenge | RGB+depth+IR CASIA-SURF solution | MIT code [GH-PARKIN] | **Code only**. Weights derive from research-only CASIA-SURF |
| SeuTao/FaceBagNet | Multi-modal CASIA-SURF winner | No license [GH-FACEBAG] | **Do not use** (all rights reserved) |
| ZitongYu/CDCN | Central-difference conv PAD | "just for research purpose" plus MIT text [GH-CDCN] | **Do not use** (research-only rider) |
| InsightFace (SCRFD, ArcFace weights) | Detector and recognizer | MIT code, **non-commercial models** [GH-INSIGHTFACE] | **Do not ship weights**. User-supplied only (as PRD says for ArcFace) |
| RealSense ID SDK | Depth face-auth module F45x | Apache-2.0 SDK [GH-RSID]; anti-spoof needs a paid license [RSID-LIC] | Depth plugin candidate only |
| librealsense | D4xx depth SDK | Apache-2.0 [GH-LIBRS] | Depth plugin candidate |
| OrbbecSDK v2 | Orbbec depth SDK | MIT [GH-ORBBEC] | Depth plugin candidate |

Not found (searched, not exhaustive): a permissively licensed, pretrained NIR-only face PAD
model, or a face detector trained on NIR. The practical route is YuNet on grey-to-3-channel IR,
hand-built cues, then a tiny classifier trained on fasec's own opt-in recordings (plan Task 6).

## 7. Commercial passive liveness (public facts only)

- **Azure Face liveness**: 0% penetration in iBeta Level 1 and 2, ISO/IEC 30107-3. Two modes.
  Passive "requires a nonbright lighting environment" and high screen brightness, about 12 s.
  Passive-Active adds an active step in bright light, up to 20 s. The orchestration is
  server-side. Mobile SDKs add RASP. The Web SDK is called "more vulnerable" [AZURE-LV],
  [IBETA-MS].
- **iProov**: "Flashmark" controlled-illumination challenge plus injection attack detection
  [IPROOV].
- **FaceTec**: "3D FaceMap" built from 100+ frames over 2-3 s of a 2D camera, with a spoof bounty
  program. These are vendor claims [FACETEC].
- **ID R&D IDLive Face**: passive, "imperceptible" to the user, with injection detection as a
  separate product (Plus) [IDRND].
- **NIST FATE PAD (NISTIR 8491)**: 82 passive software PAD algorithms on 2D imagery. Some
  algorithms handle some attack types well, none handle all [NIST-8491].
- Calibration for fasec (inferred): the vendors that reach Level 2 on RGB use screen-driven
  illumination, multi-second captures, server models and anti-injection. fasec runs locally and
  silently in the background and uses none of them. RGB-only fasec should claim "rejects casual
  photo and phone replays in good light" and nothing stronger.

## 8. Depth / 3D

- **Consumer Windows laptops with depth.** First-generation Hello devices used Intel RealSense
  F200 front depth cameras [RS-F200]. RealSense spun out of Intel in July 2025 and now targets robotics and biometrics access control, not PCs
  [RS-SPIN]. Current Hello laptops ship 2D IR, as on the dev laptop. Treat built-in depth as absent.
- **External options.** RealSense D4xx (librealsense, Apache-2.0); RealSense ID F45x, a face-auth
  module with active depth whose anti-spoofing needs a paid license [RSID]; Orbbec (SDK v2, MIT).
  Windows exposes depth sources as `MediaFrameSourceKind.Depth` through the same frame-reader API
  fasec already uses for IR [MS-FRAMES].
- **What depth would add.** Print and screen attacks fail on geometry outright. Bent-paper and
  tablet attacks stop. Some mask resistance (§2).
- **Recommendation: later, as a plugin, low priority.** Today's threat model is a photo or a
  phone held up to make fasec think the owner is present. 2D NIR with RGB fusion covers that on
  hardware users already own. Depth would serve few users and adds hardware support work. Keep
  PRD LV-5 at P2. Give the plugin interface the same `LivenessEvidence` shape as the IR tier, so
  adding depth later needs no change elsewhere. Never fake depth with monocular estimation (the
  PRD already rules it out). Never make depth a requirement.

## Open questions to verify on the dev laptop

1. **Hello concurrency.** Run `ir_probe read 2` during a Hello sign-in (Win+L, unlock) and
   during "Improve recognition". Does fasec stall, get black frames or keep its frames? Does Hello
   still succeed? (SP-7 checks 2-3, still empty.)
2. **Are unlit frames real?** Dump min/max/histogram of one unlit frame in memory, without saving
   it. All zeros means the driver blanks them and the lit−unlit cue needs another approach (lit
   frames only, or ambient subtraction). Check `FACEAUTH_MODE` support if reachable.
3. **YuNet on NIR.** Detection rate and landmark error on lit IR frames at 40-80 cm, with and
   without glasses. Is a gamma or CLAHE step needed?
4. **RGB↔IR registration.** Measure the fixed affine offset between RGB (`MI_00`) and IR
   (`MI_02`) boxes. Check it stays stable across distance (parallax).
5. **Attack bench, per species, ≥10 tries each.** Glossy and matte RGB print, phone OLED replay,
   laptop LCD replay, a printed IR capture. Record APCER per species and BPCER for the owner.
6. **Bright pupil and glint.** Are they visible at 640x360 with this emitter? With glasses?
7. **Sunlight.** Owner by a window. Does the unlit frame stay 0 (subtraction) or rise?
8. **ESS state over time.** Re-check event 1108 after the IR feature ships, to confirm fasec
   never forces a non-ESS state. Note that `bioiso` was not listed by `Get-Process`; only
   `NgcIso` was (measured).
9. **Non-ESS / non-Hello machines.** Confirm a clean fallback to the RGB tier when no IR source
   exists, or when IR open fails with `SharedReadOnly` and `ExclusiveControl` is refused.

## Sources

- [MS-HELLO-FACE] https://learn.microsoft.com/windows-hardware/design/device-experiences/windows-hello-face-authentication
- [MS-HELLO-DRV] https://learn.microsoft.com/windows-hardware/drivers/stream/windows-hello-camera-driver-bring-up-guide
- [MS-ESS] https://learn.microsoft.com/windows-hardware/design/device-experiences/windows-hello-enhanced-sign-in-security
- [MS-HELLO-SUPPORT] https://support.microsoft.com/topic/5e7c5817-54cd-468a-a00e-24699afb3f26
- [MS-ISILLUM] https://learn.microsoft.com/uwp/api/windows.media.capture.frames.infraredmediaframe.isilluminated
- [MS-FRAMES] https://learn.microsoft.com/uwp/api/windows.media.capture.frames.mediaframesourcekind
- [CVE-2025-26644] https://msrc.microsoft.com/update-guide/vulnerability/CVE-2025-26644 ; reporting: https://heise.de/-10449393
- [TH-DARK] https://www.tomshardware.com/uk/software/windows/microsoft-has-broken-windows-hello-facial-recognition-it-no-longer-works-in-the-dark
- [RED-BLEED] https://www.usenix.org/conference/usenixsecurity25/presentation/hu-bowen
- [CYBERARK] https://www.cyberark.com/resources/threat-research-blog/bypassing-windows-hello-without-masks-or-plastic-surgery
- [KB5005478] https://support.microsoft.com/en-US/servicing/os/windows/2021/07/kb5005478-windows-hello-cve-2021-34466
- [APPLE-SEC] https://support.apple.com/guide/security/face-id-and-touch-id-security-sec067eb0c9e/web
- [APPLE-FACEID] https://support.apple.com/en-us/102381
- [AOSP-MEASURE] https://source.android.com/docs/security/features/biometric/measure
- [PIXEL8] https://www.androidauthority.com/google-pixel-8-face-unlock-3370970 ; https://www.androidcentral.com/phones/google-pixel-8-face-unlock-upgrade
- [ISO-30107-3] https://www.iso.org/standard/79520.html
- [PAD-METRICS] https://arxiv.org/pdf/2010.04145
- [IBETA] https://www.ibeta.com/iso-30107-3-presentation-attack-detection-confirmation-letters/
- [IBETA-MS] https://www.ibeta.com/wp-content/uploads/2025/04/250414-Microsoft-PAD-Retest-Confirmation-Letter.pdf
- [FIDO-BIO] https://fidoalliance.org/specs/biometric/requirements/Biometrics-Requirements-v4.1-fd-20250106.html
- [NIST-8491] https://nvlpubs.nist.gov/nistpubs/ir/2023/NIST.IR.8491.pdf ; https://pages.nist.gov/frvt/html/frvt_pad.html
- [AZURE-LV] https://learn.microsoft.com/azure/ai-services/face/concept-face-liveness-detection
- [IPROOV] https://www.iproov.com/llms.txt
- [FACETEC] https://www.facetec.com/security
- [IDRND] https://www.sourceforge.net/software/compare/IDLive-Face-vs-IDLive-Face-Plus/
- [NIR-REFL] https://www.mdpi.com/2076-3417/13/3/1987
- [NIR-SKIN] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11053657/
- [BRIGHT-PUPIL] https://openaccess.thecvf.com/content_iccv_2015_workshops/w12/papers/Borsato_A_Fast_and_ICCV_2015_paper.pdf
- [IBM-PUPIL] https://research.ibm.com/publications/differences-in-the-infrared-bright-pupil-response-of-human-eyes
- [HQ-WMCA-SWIR] https://arxiv.org/pdf/2007.11469
- [CASIA-SURF] https://arxiv.org/abs/1812.00408
- [CEFA-DL] https://sites.google.com/view/face-anti-spoofing-challenge/dataset-download/casia-surf-cefacvpr2020
- [WMCA] https://www.idiap.ch/dataset/wmca ; paper https://arxiv.org/pdf/1909.08848
- [WMCA-ZENODO] https://zenodo.org/records/4580313
- [HQ-WMCA] https://www.idiap.ch/dataset/hq-wmca
- [YUNET-LIC] https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE
- [SFACE-LIC] https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE
- [GH-SILENT] https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- [GH-YAKHYO] https://github.com/yakhyo/face-anti-spoofing
- [GH-MEDIAPIPE] https://github.com/google-ai-edge/mediapipe
- [GH-KPROKOFI] https://github.com/kprokofi/light-weight-face-anti-spoofing
- [GH-PARKIN] https://github.com/AlexanderParkin/ChaLearn_liveness_challenge
- [GH-FACEBAG] https://github.com/SeuTao/FaceBagNet
- [GH-CDCN] https://github.com/ZitongYu/CDCN/blob/master/LICENSE
- [GH-INSIGHTFACE] https://github.com/deepinsight/insightface#license
- [GH-RSID] https://github.com/realsenseai/RealSenseID
- [RSID] https://www.realsenseai.com/id-pro ; [RSID-LIC] https://www.tecisoft.com/products/intel-realsense-id-anti-spoofing-license-1-device-3dsf450asl1
- [GH-LIBRS] https://github.com/realsenseai/librealsense
- [GH-ORBBEC] https://github.com/orbbec/OrbbecSDK_v2
- [RS-F200] https://windowscentral.com/video-windows-hello-realsense-camera
- [RS-SPIN] https://techcrunch.com/2025/07/11/realsense-spins-out-of-intel-to-scale-its-stereoscopic-imaging-technology
