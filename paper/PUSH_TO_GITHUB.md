Steps to push this repository to your GitHub account and trigger the CI build for the IEEE PDF:

1. Create a new GitHub repository in your account (e.g., `train-delay-prediction-paper`).
2. In your local repo root, add the remote and push the current branch:

```powershell
# add your repo; replace <username> and <repo>
git remote add origin git@github.com:<username>/train-delay-prediction-paper.git
# or use HTTPS
# git remote add origin https://github.com/<username>/train-delay-prediction-paper.git

git add .
git commit -m "paper: add IEEE LaTeX sources, figures, and CI to build PDF"
git push -u origin HEAD
```

3. After pushing, go to the repository on GitHub -> Actions. The `build-paper` workflow will run (it installs TeX Live and builds `paper/main.pdf`). When the workflow completes, download the artifact `paper-main-pdf` from the workflow run page.

4. If you want the PDF built locally instead of via Actions, you can run the included Docker build (requires Docker installed):

```powershell
# from repo root
cd paper
# build the Docker image
docker build -t train-delay-paper .
# run it; the built PDF will be in paper/ output inside the container and mapped to the host
docker run --rm -v ${PWD}:/work train-delay-paper
# the script writes paper/main.pdf to the host folder
```

Notes:
- Ensure GitHub Actions is enabled on your repository and that workflows are allowed to run.
- If your repo is private, Actions artifacts are still accessible to repository members.
- If you want me to prepare a draft PR with finalized author info and commit message, tell me and I will prepare the branch and the exact commands for you to run or grant permission for me to push (you must add my collaborator account or run the commands locally).