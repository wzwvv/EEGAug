# EEG Data Augmentation

Welcome! This repo studies data augmentation approaches for EEG analysis, specifically for brain-computer interface (BCI) applications.
The official implementation of Channel Reflection (CR), of paper [`Channel Reflection: Knowledge-Driven Data Augmentation for EEG-Based Brain-Computer Interfaces`](https://www.sciencedirect.com/science/article/pii/S0893608024002752) (**Neural Networks, 2024**)

## Steps for Usage:

#### Install Dependencies

Install dependencies based on  `environment.yml` file.

#### Run it!

We have already provided the processed data of BNCI2014001 of MOABB (details see paper) under ./data/
To use other datasets, follow a similar format.

Run
```sh 
python within_baseline.py
```   
or   
```sh 
python within_CR.py
```  

for comparison of with or without CR results for within-subject classification.

Results are shown in the following table:

| Approach  |  5  |  10  |  15  |  20  | 25  |  30  |  35  |  40  |  45  |  Avg.  |
|:---------:|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Baseline  | 52.49 | 65.77 | 67.54 | 67.63 | 68.20 | 70.50 | 70.87 | 74.48 | 74.07 | 67.95 | 
|    CR     | 61.36 | 64.34 | 69.59 | 70.73 | 72.81 | 72.22 | 75.08 | 77.60 | 75.93 | 71.07 |

## Contact

Please use Issues for any questions regarding the code, or contact me at vivi@hust.edu.cn for any questions regarding the paper.

## Citation

If you find this repo helpful, please cite our work:
```
@article{Wang2024CR,
  title={Channel Reflection: Knowledge-driven data augmentation for {EEG}-based brain-computer interfaces},
  author={Wang, Ziwei and Li, Siyang and Luo, Jingwei and Liu, Jiajing and Wu, Dongrui},
  journal={Neural Networks},
  pages={106351},
  year={2024}
}
```

## More Regarding Neural Networks
If you are interested in neural network based models, do check out [`Deep Transfer for EEG`](https://github.com/sylyoung/DeepTransferEEG) 