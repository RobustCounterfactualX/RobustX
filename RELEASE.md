"""
## Publishing a New Version to PyPI

This project uses **GitHub Actions** to automatically build and publish a new version of the package to [PyPI](https://pypi.org/) whenever a versioned **Git tag** is pushed.

### Steps to Release a New Version

1. **Commit & Push Any Changes**
   ```sh
   git add .
   git commit -m "Prepare for vX.Y.Z release"
   git push origin main  # or your development branch
   ```

2 **Create a New Version Tag**
   Replace `X.Y.Z` with the new version number:
   ```sh
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
   **Example:**
   ```sh
   git tag v1.2.3
   git push origin v1.2.3
   ```

3. **GitHub Action Automatically Builds & Publishes**
   - ✅ The workflow **builds the package**.
   - ✅ It **publishes to PyPI** using **trusted publishing**.
   - ✅ A **GitHub release** is created with signed artifacts.

4. **Verify the Release on PyPI**
   - Check the new version on PyPI:  
     🔗 [https://pypi.org/project/RoCELib/](https://pypi.org/project/RoCELib/)

5. **Test the New Version**
   Once the package is live, install and verify it:
   ```sh
   pip install --upgrade RoCELib
   ```

### \U0001F4CC Notes
- **Tag-Based Publishing:** Only **pushed tags** (e.g., `v1.2.3`) trigger a release. Commits to `main` **do not** automatically publish.
- **Test Before Release:** Check the results of the tests CI pipeline before creating a tag for a commit.

🚀 **You're all set!** Just tag and push to publish a new version! 🎉
"""