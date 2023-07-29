# tsunami_gauges_optimization
Python codes for optimizing the configuration of tsunami gauges and reconstructing wave height distribution. 

```
.
├── README.md
├── input
│   ├── gauge.csv                   Input file: Synthetic gauge information
│   └── quake.csv                   Input file: Fault rupture parameters of hypothetical earthquake scenarios
└── programs
    ├── pre-process.py              Main code: Data loading and proper orthogonal decomposition
    ├── optimization.py             Main code: Optimization of the gauge arrangement
    ├── pseudo-super-resolution.py  Main code: Reconstruction of wave height distribution
    ├── subscript.py                Module
    └── inverse_problem.py          Module: Solver of inverse problem 
```

## Environment
Tested and confirmed compatible environments
- Ubuntu 20.04.6 LTS
- python 3.8.10

## Required python libraries
- numpy==1.22.1
- pandas==1.3.5
- scipy==1.8.1
- dask==2022.05.0

