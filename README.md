---

# Message Passing Optimization-Based Constrained Learning (MPOCtrL)

This project provides tools to run computational tasks on an HPC cluster managed by SLURM or on your local machine. Follow the instructions below to get started.

---

## **Prerequisites**

### 1. Install Miniconda
If you do not have Miniconda installed:
1. Go to the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html).
2. Download and install the appropriate version for your operating system.
3. Follow the installation instructions for your operating system.

---

## **Setup**

### 2. Create a Virtual Python Environment
This project uses a `environment_hpc.yml` (`yml` not `txt`) file to set up the Python environment with all necessary libraries and dependencies.

1. Open your terminal or command prompt.
2. Navigate to the project directory (where this `README.md` is located).
3. Run the following commands:

   ```bash
   conda env create -f environment_hpc.yml
   ```

   This will automatically create a Conda environment with the name specified in `environment_hpc.yml`. If you need a custom environment name, run:

   ```bash
   # conda env create -f environment_hpc.yml --name <your_env_name>
   ```

4. Activate the newly created environment:

   ```bash
   conda activate mpoctrl_env
   ```

   Replace `<env_name>` with the environment name specified in `requirements.yml` or the one you provided.

5. Test your environment
   ```bash
   python test_env.py # should output "All libraries are installed correctly"
   ```

---

## **How to Use**

### **Option 1: Submit Jobs to an HPC**

The HPC is assumed to be managed by SLURM. Follow these steps to submit your jobs:

#### 1. Edit the Python Job Submission Script
Choose the appropriate Python script based on your HPC's GPU availability:
- For GPU-enabled HPC: `submit_jobs_to_hpc_gpu.py`
- For CPU-only HPC: `submit_jobs_to_hpc_cpu.py`

Open the respective script and update the following parameters:
- **Input Data Directory**: Path to your input data folder.
- **Output Directory**: Path to the folder where results will be saved.
- **Data Names**: Names of the data files or datasets.
- **Email Address**: Your email address to receive job notifications.
- ** other parameters as needed

#### 2. Submit the Job
Run the appropriate Python script to submit the job:
```bash
python submit_jobs_to_hpc_gpu.py
```
or
```bash
# python submit_jobs_to_hpc_cpu.py
```

#### 3. Job Workflow
The job submission workflow involves the following:
1. `submit_jobs_to_hpc_gpu.py` or `submit_jobs_to_hpc_cpu.py`: Prepares and submits your job to SLURM.
2. `run_on_hpc_gpu.sh` or `run_on_hpc_cpu.sh`: Executes the task on the HPC, passing data-related parameters to the main program.
3. `src/main.py`: The main Python file where the algorithm runs.

---

### **Option 2: Run the Code Locally**

If you want to run the code on your local laptop:

#### 1. Edit the Local Run Script
Open the shell script `run_on_local.sh` and update the following:
- **Input Data Directory**: Path to your input data folder.
- **Output Directory**: Path to the folder where results will be saved.
- **Data Names**: Names of the data files or datasets.

#### 2. Run the Script
Execute the shell script using the following command:
```bash
sh run_on_local.sh
```

---

## **Output**
The algorithm will generate an output results directory with a name in the format:

```
<data_name>-<network_name>-Flux-<time>
```

You will find all result files in this directory. The structure includes:
- Computation results
- Logs
- Any other generated files

---

## **Summary**
- **HPC Workflow**: `submit_jobs_to_hpc_gpu.py` (or `submit_jobs_to_hpc_cpu.py`) → `run_on_hpc_gpu.sh` (or `run_on_hpc_cpu.sh`) → `src/main.py`
- **Local Workflow**: `sh run_on_local.sh`

Feel free to reach out if you encounter any issues!
