# Third-party software and model notices

This service is an API adapter; the biometric models and upstream engine are not authored here.

- **OpenBiometrics engine and API**: https://github.com/openbm/openbiometrics, pinned to commit `ab09d469327684d630b7383119dd4544253a03c9`. The upstream repository includes an MIT license. Its copyright/license notice must be preserved.
- **Community face models**: the pinned OpenBiometrics registry labels the required YuNet detector as MIT, the SFace recognizer as Apache-2.0, and its MiniFASNet anti-spoof model as Apache-2.0. The Docker build requests only these three model assets and deliberately skips the optional age/gender model. Verify the model registry and source-artifact terms for the exact pinned build before production; repository-level licensing does not automatically license every third-party model file.
- No legacy InsightFace model group is requested by this Docker build. Do not switch to legacy/premium model artifacts without reviewing their license and redistribution/commercial-use terms.

The upstream engine and its model artifacts are provided without a guarantee that they satisfy the required accuracy, fairness, anti-spoofing, privacy, legal, or regulatory standards for a real attendance deployment.
