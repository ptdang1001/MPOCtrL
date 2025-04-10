
---
# Message Passing Optimization-Based Constrained Learning (MPOCtrL)

This project provides tools to run computational tasks on an HPC cluster managed by SLURM or on your local machine. Follow the instructions below to get started.

---

## **Prerequisites**

### 1. Install Miniconda
If you have Miniconda installed:
1. Some Slurm - Linux servers come with Miniconda3 or Conda pre-installed. If you need to activate your Conda environment, please contact your IT support team for assistance. 


If you do not have Miniconda installed:
1. Go to the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html).
2. Download and install the appropriate version for your operating system.
3. Follow the installation instructions for your operating system.
4. If you lack administrator or sudo privileges on the server, you might need to install a local version of Miniconda3. In this case, you'll likely need to adjust your environment's PATH variable to point to your local Miniconda3 installation. You can find numerous online tutorials that explain how to do this.

    ```bash
    # run the command to make sure conda works
    conda --version
    ```

---

## **Setup**

### 2. Create a Virtual Python Environment
This project uses a `requirements_hpc.yml` (`yml` not `txt`) file to set up the Python environment with all necessary libraries and dependencies.

1. Open your terminal or command prompt.
2. Navigate to the project directory (where this `README.md` is located).
3. Run the following commands:

   ```bash
   conda env create -f requirements_hpc.yml
   ```

   This will automatically create a Conda environment with the name specified in `requirements_hpc.yml`. If you need a custom environment name, run:

   ```bash
   # conda env create -f requirements_hpc.yml --name <your_env_name>
   ```

4. Activate the newly created environment:

   ```bash
   conda activate mpoctrl_env
   ```

   Replace `<env_name>` with the environment name specified in `requirements_hpc.yml` or the one you provided.

5. Test your environment

   ```bash
   python test_env.py # should output "All libraries are installed correctly"
   ```

---

## **Data Pre-processing**

Before running the MPOCtrL algorithm, you may need to pre-process your data (TPM/CPM normalized, not counts data) to handle sparsity. We provide three approaches:

`src/tools/counts_2_cpm.py`
`src/tools/filter_dense.py`
...

    ```bash
    cd src/tools/
    ```

## Option 1: Counts data --> CPM/TPM normalized data (`counts_2_cpm.py`)
- Normalize the Counts to CPM/TPM data
- Inputs: 
-- data_dir: /path/to/your_data/
-- data_file: data_name.csv/data_name.csv.gz (rows:= genes, columns:= samples/cells)
- Outputs:
-- data_name_cpm.csv.gz (rows:= genes, columns:= samples/cells), default saved in `data_dir`

To run data normalization:

    ```bash
    # counts --> cpm/tpm, choose command 1 or or or or or command 2
    # command 1 (Recommend, if your data is small): 
    python counts_2_cpm.py --data_dir /path/to/your_data/ --data_name data_name.csv.gz

    # Or Or Or Or

    # command 2 (Recommend if your data is large): 
    # modify the dta_dir and data_name in the shell script please, 
    # submit the job to server. No need to stare at the laptop screen and wait.
    sbatch counts_2_cpm_sbatch.sh
    ```

### Option 2: Filter Dense Samples (`filter_dense.py`)
- Removes sparse samples from your dataset
- Preserves only samples with sufficient data density
- Recommended when you have enough samples and want to work with high-quality data
- Inputs: 
-- data_dir: /path/to/your_data/
-- data_file: data_name.csv/data_name.csv.gz (rows:= genes, columns:= samples/cells)
-- percentages: "5,10,20" # means top dense 5%, 10%, 20%, ... samples

- Outputs:
-- data_name_gene123_sample456.csv.gz (rows:= genes, columns:= samples/cells), default saved in `data_dir`

    ```bash
    # For filtering dense samples, command 1 or or or or or command 2
    # command 1 (Recommend, if your data is small): 
    python filter_dense.py --data_dir /path/to/your_data/ --data_name data_name.csv.gz

    # or or or or or 

    # # command 2 (Recommend if your data is large): 
    # modify the dta_dir and data_name in the shell script please, 
    # submit the job to server. No need to stare at the laptop screen and wait.
    sbatch filter_dense_sbatch.sh
    ```


### Option 3: Gene Imputation (`gene_imputation.py`)
- Predicts missing values using advanced imputation techniques, default NMF
- Recommended when you need to preserve all samples but want to handle missing data
- Note: Gene imputation for sparse data remains challenging in ML/AI fields
- Inputs: 
-- data_dir: /path/to/your_data/
-- data_file: data_name.csv/data_name.csv.gz (rows:= genes, columns:= samples/cells)

- Outputs:
-- data_name_imputed_rank5.csv.gz (rows:= genes, columns:= samples/cells), default saved in `data_dir`

    ```bash
    # For gene imputation
    # command 1 (Recommend, if your data is small):
    python gene_imputation.py --data_dir /path/to/your_data/ --data_name data_name.csv.gz

    # or or or or or 

    # # command 2 (Recommend if your data is large): 
    # modify the dta_dir and data_name in the shell script please, 
    # submit the job to server. No need to stare at the laptop screen and wait.
    sbatch gene_imputation_sbatch.sh
    ```

### Recommendations:
1. **Try both approaches (option 2 or option 3) separately** - There's no perfect solution for sparse data
2. **GPU Acceleration** - The conda environment/algorithm is optimized for CUDA GPU servers
3. **Run on HPC** - For best performance, use a SLURM-managed Linux CUDA GPU server
4. **Not tested** on macOS/Windows/Linux laptops - Server execution is strongly recommended

---

## **How to Use MPOCtrL**

### **Option 1: Submit Jobs to an HPC**

The HPC is assumed to be managed by SLURM, CUDA GPU. Follow these steps to submit your jobs:

#### 1. Edit the Python Job Submission Script
Choose the appropriate Python script based on your HPC's GPU availability:
- For GPU-enabled HPC: `submit_jobs_to_hpc_gpu.py`
- For CPU-only HPC: `submit_jobs_to_hpc_cpu.py`

Open the respective script and update the following parameters:
- **Input Data Directory**: Path to your input data folder: /path/to/your data/ (use pre-processed data if applicable)
- **Output Directory**: Path to the folder where results will be saved: /path/to/your results/
- **Data Names**: Names of the data files or datasets: data_name.csv/data_name.csv.gz (rows:= genes, columns:= samples/cells)
- **Email Address**: Your email address to receive job notifications: xxxxx@xxx.com/xxxxx@xxxx.edu 
- **Other parameters** as needed

#### 2. Submit the Job
Run the appropriate Python script to submit the job:
```bash
python submit_jobs_to_hpc_gpu.py # Recommend
```
or
```bash
# python submit_jobs_to_hpc_cpu.py
```

#### 3. Job Workflow
The job submission workflow involves the following:
1. `submit_jobs_to_hpc_gpu.py` or `submit_jobs_to_hpc_cpu.py`: Prepares and submits your job to SLURM.
-->
2. `run_on_hpc_gpu.sh` or `run_on_hpc_cpu.sh`: Executes the task on the HPC, passing data-related parameters to the main program.
-->
3. `src/main.py`: The main Python file where the algorithm runs.

---

### **Option 2: Run the Code Locally**

If you want to run the code on your local laptop:

#### 1. Edit the Local Run Script
Open the shell script `run_on_local.sh` and update the following:
- **Input Data Directory**: Path to your input data folder (use pre-processed data if applicable)
- **Output Directory**: Path to the folder where results will be saved
- **Data Names**: Names of the data files or datasets

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
- **Pre-processing Options**: `counts_2_cpm.py`,  `filter_dense.py` or `gene_imputation.py`
- **HPC Workflow**: `submit_jobs_to_hpc_gpu.py` → `run_on_hpc_gpu.sh` → `src/main.py`
- **Local Workflow**: `sh run_on_local.sh`

> **Note on Performance**: For optimal performance with large datasets, always run on a GPU-enabled server. Local execution on laptops is not recommended for production runs.
