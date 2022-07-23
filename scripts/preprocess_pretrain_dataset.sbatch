#!/bin/bash
#SBATCH --job-name=preprocess_pretrain_dataset
#SBATCH --error=preprocess_pretrain_dataset-%j.err        # Файл для вывода ошибок 
#SBATCH --output=preprocess_pretrain_dataset-%j.log       # Файл для вывода результатов 
#SBATCH --time=00:00:30                      # Максимальное время выполнения 
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу 
#SBATCH --gpus=4                            # Требуемое кол-во GPU


module purge
module load Python/Anaconda_v10.2019

source deactivate
source activate KPGT

which python
python -V

python preprocess_downstream_dataset.py --data_path ../datasets/ --dataset bace --n_jobs 4

