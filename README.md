# TORQUE-dataset

This repository contains the TORQUE dataset published in the following paper:
```
@inproceedings{NWHPGR20,
  title = {{TORQUE}: A Reading Comprehension Dataset of Temporal Ordering Questions},
  author = {Ning, Qiang and Wu, Hao and Han, Rujun and Peng, Nanyun and Gardner, Matt and Roth, Dan},
  booktitle = {EMNLP},
  year = {2020}
}
```
The main website hosting all relevant information about this paper can be found [here](https://allennlp.org/torque.html).

## Description
- `./data` contains the entire dataset and our original train/dev/test split. Note we have removed all the annotations in the test set. For evaluation, please visit our leaderboard. In addition, we have also included `question_clustering.json` that describes the clusters of contrast questions.
- `./basic_stats` contains the script `basic_stats.py` that produces Table 3, Figure 12, Figure 13, Figure 15, and Figure 16. Please note that those tables and figures in the paper were generated on the entire dataset, while the ones generated here are only on the training set, so you may expect some minor mismatches.
- `./Table2_sample_questions_categorization` contains the original spreadsheet the author used to categorize all these questions, and a jupyter notebook that generates Table 2.
- `./Fig11_learning_curve` contains a jupyter notebook plotting Figure 11.