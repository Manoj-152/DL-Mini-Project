# DL-Srping 2024 Mini Project #

Submission by Manoj Srinivasan, Xiaoxiao Shi, Pengbo Wang

We achieved our highest score of 0.815 on the public leaderboard using the best_ckpt_student_distil.pth (using student-teacher distillation). The evaluate.ipynb code can be used to generate the submission csv file and produce example predictions.

## Code to run the evaluation script

``` 
python3 evaluate.py --model_ckpt <PATH TO MODEL CKPT> --dataset_path <PATH TO TEST DATASET>
``` 

## Code to run the distillation training
(Make sure the teacher is already trained and checkpoint is stored (already included with the repo))
``` 
python3 train_distillation.py
``` 

## Code to train the teacher
``` 
python3 train_teacher.py
``` 

## Code to run the augmentation-only training (no distillation)
```
python3 train.py
```
