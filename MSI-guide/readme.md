
# Two Common Ways to Login to the MSI Server

- **MSI Introduction:** [MSI Website](https://msi.umn.edu/)
- **Important:** In both approaches, you must install Cisco Secure Client and connect to the VPN before connecting to MSI.

## Approach 1: MSI Ondemand, an Interactive GUI

- **Login:** [MSI Ondemand](https://msi.umn.edu/about-msi-services/interactive-hpc/open-ondemand)
- **Terminal Access:** Navigate to the "Cluster" tab and select "Agate Shell Access" for terminal interaction.
- **Python Environment:** It's recommended to create a virtual environment, such as "GenAI", before starting your work. Refer to the "Set up Python" section below.
- **Jupyter Notebook Setup:** Select the "My Interactive Sessions" tab and launch "Jupyter". Configure your session with the following settings:
  - **Jupyter Interface:** Lab
  - **Jupyter Python:** Custom
  - **Custom Python Environment:** 
    ```
    module load cuda
    module load conda
    conda activate GenAI  # Optional, if "GenAI" was set up previously
    ```
  - **Notebook Root:** (Leave empty or set your preferred directory)
  - **Account:** (Use an accessible account, e.g., stat8931)
  - **Resources:** Select "interactive GPU ... 1 A40" for GPU needs, or "Custom" for specific resources.
  - **Partitions:** Choose between "a100-4" or "a100-8" for general GPU access or "Custom" for PI-specific resources.
  - **Custom Partitions:** For example, "jd-4a100" for JieGroup.
  - **Number of Nodes:** 1 (Adjust only if necessary for node parallelism)
  - **Cores per Node:** 8
  - **Memory per Node:** 8192 MB
  - **Scratch per Node:** 0-500 GB
  - **GPUs per Node:** 1 (Adjust only if necessary for GPU parallelism)
  - **Time Limit:** No more than 24 hours
  - **Email Notification:** Opt-in to receive an email when the session starts.

## Approach 2: Use VS Code

- **Setup:** Install the Remote - SSH plugin from VSCode extensions.
- **Connection:** Open the "Remote Explorer", type `ssh umnID@agate.msi.umn.edu` to connect.
- **Authentication:** Enter your UMN password and respond to the Duo prompt to login.
- **VSCode Integration:** Use the terminal and file navigation within VSCode for interaction.

### Submitting Jobs to MSI

- **Job Submission:** Unlike Ondemand, you must submit a PBS file (`main.pbs`) which schedules the job for CPU/GPU resource utilization.
  - Update the python environment path in `main.pbs`. Use `echo $CONDA_PREFIX` to verify the path on your server.
  - Replace `COMMAND_PLACEHOLDER` with a python command, e.g., `python test.py`.
- **Concurrent Jobs:** Use `submit_jobs.py` to manage multiple job submissions. Exercise caution to avoid system overload.

### GPU Status Check

- **Command:** `!nvidia-smi` to view available GPU resources.

## Set up Python

- **Environment Creation:** 
  ```bash
  conda create -n GenAI python=3.8 ipython
  ```

- Add Environment to Jupyter:
  ```bash
  conda install jupyter ipykernel
  python -m ipykernel install --user --name=GenAI --display-name="Python 3.8 (GenAI)"
  ```

- Install Packages:
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

- (optional) use the following to export installed packages from another environment to the current one
  ```bash
  pip freeze > requirements.txt
  pip install -r requirements.txt
  ```
