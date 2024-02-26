# EMBARK project

This repository provides the code and example data of the paper "Efficient Modeling Attack on MPUFs via Kronecker Matrix Multiplication"

## Usage
To run the default **EMBARK** attack process, simply excute 

    python embark.py
    
To run **EMBARK+COMB**, excute

    python integrated_embark.py

Our program will automatically generate CRPs, and then conduct attack process. If you need to import training and testing datasets separately from external files, uncomment the lines with `loadTrainData` and `loadTestData`.

    # train_loader = loadTrainData("./data/5_64_MPUF_50k_train.csv", config.batch_size, config.device)
    # valid_loader, test_loader = loadTestData("./data/5_64_MPUF_50k_test.csv", config.batch_size, config.device)

## Configuration

### PUF setting and dataset
To conduct attack for $(n,k)$-MPUF, you only need to specify the data filename in the very first line of the main function in `embark.py` and `integrated_embark.py`. Our program will automatically prase the arguments in the filename. 

The filename should be formatted like `{k}_{n}_{PUF type}_{data size}.csv`. For example, `5_64_MPUF_50k.csv` leads to an attack for $(64, 5)$-MPUF with a dataset including 50k CRPs. For noisy environments, format filename like ``{k}_{n}_{PUF type}_{a}_{data size}.csv``. The argument `a` indicates a noise level of $1/a$.

### Hyperparameter
The following hyperparameters can be set in `/utils/config.py`.

- learning parameters: learning rate, batch size, loss function, etc.
- algorithm parameters: the coefficients of integrated EMBARK loss.

See `/utils/config.py` for more specifices.

## Model
`/utils/model.py` provides the implementation of models includeing

- PUF models: MPUF, cMPUF, rMPUF models used to generate CRPs
- attack models: EMABRK and integrated EMBARK models for MPUF, cMPUF and rMPUF