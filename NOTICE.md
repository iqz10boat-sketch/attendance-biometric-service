# Third-party software and model notices

This service is an API adapter; the biometric models and upstream engine are not authored here.

- **OpenBiometrics engine and API**: https://github.com/openbm/openbiometrics, pinned to commit `ab09d469327684d630b7383119dd4544253a03c9`. The upstream repository includes an MIT license. Preserve its copyright and license notices.
- **YuNet detector**: the pinned OpenBiometrics model catalog labels this model MIT-licensed. The Docker build retains the pinned catalog URL and metadata.
- **SFace recognizer**: the pinned OpenBiometrics model catalog labels this model Apache-2.0. The Docker build retains the pinned catalog URL and metadata.
- **Passive anti-spoofing model**: `facenox/face-antispoof-onnx`, release `v1.0.0`, asset `best_model.onnx`, downloaded from https://github.com/facenox/face-antispoof-onnx/releases/download/v1.0.0/best_model.onnx and pinned by SHA-256 `af2381b88f38769222ed93379e12444e2a50814575de1c46170de570c55a42b6`. The source repository provides an Apache-2.0 license and says the model was trained on CelebA-Spoof. Review the dataset/model provenance and permitted intended use before production use.

The model publisher's reported accuracy is not an independent PAD certification and does not guarantee detection of all printed photos, screen replays, masks, or other attacks. A passing health check only verifies initialization, not accuracy, fairness, or robustness. The service must fail closed if liveness is unavailable, and the model must pass a supervised, documented acceptance test before real student attendance is enabled.
