# ScaffoldInvent

ScaffoldInvent is a deep learning project for molecular generation and scaffold-based molecular sampling, designed for cheminformatics and drug discovery research. 

## Features

- Automatic extraction and processing of molecular scaffolds
- Support for multiple scaffold extraction methods, including classic methods and an MMP (Matched Molecular Pair) approach
- Molecular generation and sampling
- Evaluation and filtering of molecular properties
- Model training and fine-tuning support
- Supports multiple molecular data formats (e.g., SMILES)

## Repository Structure

- `Train.py`: Model training
- `sample.py`, `sample_active.py`: Molecular sampling and generation
- `Sca_extraction.py`: Scaffold extraction utilities (classic scaffold extraction)
- `mmp.py`: MMP scaffold extraction tool (Matched Molecular Pair method)
- `fine_tuning.py`: Model fine-tuning scripts
- `Metrics/cal_metric_active_5k.py`: Evaluation of generated molecules

## How to Run

1. **Install dependencies**

   Make sure you have Python installed, and install the following key dependencies (refer to your requirements.txt for details):

   ```bash
   conda env create -f environment.yml
   ```

2. **Prepare datasets and pretrained models**

   - Dataset files in SMILES format (e.g., `data/*.smi`)
   - Pretrained model files (e.g., `model/*.ckpt`)
   - Scaffold vocabulary files (e.g., `data/Voc_merged_chembl_all`)

3. **Run scripts**

   - **Molecule sampling/generation**:

     ```bash
     python sample.py --vocPath ./data/Voc_merged_chembl_all --modelPath ./model/merged_chembl_pretrain_attention.ckpt --save_dir ./data/sample_output.csv --datasetPath ./data/your_input.smi --batch-size 32 --epochs 400 --molecule_num 100
     ```

   - **Model training**:

     ```bash
     python Train.py --datasetPath ./data/train.smi --voc_dir ./data/voc.txt --batch-size 32 --epochs 100 --hidden_size 256
     ```

   - **Scaffold extraction (classic method)**:

     ```bash
     python Sca_extraction.py --data ./data/input.smi --save_dir ./data/output_sca.smi
     ```

   - **Scaffold extraction (MMP method)**:

     ```bash
     python mmp.py --data ./data/input.smi --save_dir ./data/output_mmp_sca.smi
     ```

   - **Evaluation of generated molecules**:

     ```bash
     python Metrics/cal_metric_active_5k.py --train_path ./data/train.csv --gen_path ./data/sample_output.csv --output ./data/metrics.csv
     ```

4. **Custom parameters**

   Each script supports customizable command-line arguments. Refer to the argparse settings inside each script, or run `python <script>.py --help` for details.

## Notes

- Recommended to run in a Linux environment; GPU acceleration (CUDA) is suggested.
- Some default paths are set as `/ScaffoldInvent/`, please update them to your own data/model paths as needed.
- If you have questions or need further customization, please refer to the script comments or contact the project author.

---

If you need more detailed documentation or code examples, feel free to ask!