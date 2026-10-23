# Publish this source to your empty GitHub repository (Windows PowerShell)

The GitHub connection in ChatGPT can read the repository metadata but its write API returned `Resource not accessible by integration`. The repository itself is still empty. Until the connector's write permission is fixed, these commands publish the prepared source from your own machine using the Git credentials already configured there.

1. Download and extract `attendance-biometric-service-source.zip`.
2. Open PowerShell in the extracted `attendance-biometric-service` folder.
3. Run:

```powershell
git init -b main
git add .
git commit -m "Build authenticated OpenBiometrics adapter for Render"
git remote add origin https://github.com/iqz10boat-sketch/attendance-biometric-service.git
git push -u origin main
```

Do not put passwords or personal access tokens in these commands. Git should use your configured credential manager/browser sign-in. If Git reports that `origin` already exists, run `git remote -v` and update it with `git remote set-url origin https://github.com/iqz10boat-sketch/attendance-biometric-service.git`.

After push, open **Actions** and inspect the **Biometric service CI** run. The workflow tests the adapter and builds the Docker image, including fetching the pinned upstream code and model files. A successful mock/unit test alone does not count as a model-backed or Render deployment pass.
