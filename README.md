# FLASHE

This repository contains scripts and instructions for reproducing the experiments in our **arXiv '21** paper [FLASHE](#).
FLASHE is integrated as a pluggable module in [FATE v1.2.0](https://github.com/FederatedAI/FATE/tree/v1.2.0), an open-source industrial platform crafted by [WeBank](https://www.webank.com/#/home) for cross-silo federated learning.
You may want to use commands like `git diff` to see what have been changed since the fork (i.e., the [first commit](https://github.com/SamuelGong/FLASHE/commit/bacce3035f5972d4ec3f59e42c14152090664895)).

# Overview

* [Run Experiments and Validate Results](#run-experiments-and-validate-results)
* [Repo Structure](#repo-structure)
* [Acknowledgement](#acknowledgement)
* [Contact](#contact)

# Run Experiments and Validate Results

The output of the experiment will validate the following major claims in our background part (section 2 in paper) and evaluation part (section 6 in paper):

####    **Cluster Deployment**
1. FLASHE outperforms batching versions of the three baselines by 3.2×-15.1× in iteration time and 2.1×-42.4× in network footprint. Compared to plaintext training, FLASHE achieves near-optimality efficiency with overhead ≤6% and 0% in traffic (§6.2) —> Figure 7 and 8.
2. FLASHE brings down at least 13×-63× of the computation overhead and 48× of the communication overhead of general HEs when sparsification is employed (§6.4) —> Figure 10 and 11.
3. FLASHE's mask precomputation effectively eliminates the cryptographic overhead (rendering it < 0.1s) (§6.6) —> Figure 13.

####    **Local Simulation and Projection**
1. The enormous performance of general HE schemes like Paillier, FV, and CKKS raises significant efficiency and scalability concerns. Even with the aid of batch encryption, the message inflation factor remains suboptimal (§2.2 and §2.3) —> Table 2.
2. FLASHE exhibits near-optimality when it comes to economic cost (≤5%). Compared with batching versions of the baselines, the savings are significant (up to 73%-94%) (§6.3) —> Figure 9 and Table 4.
3. FLASHE's double masking scheme achieves optimal latency when the client dropout is mild (§6.5) —> Figure 12.

## Cluster Deployment

Please go to `./deployment` directory and follow the deployment [README](./deployment/README.md) to run related scripts.

## Local Simulation and Projection

Please go to `./simulation` directory and follow the simulation [README](./simulation/README.md) to run related scripts.

# Repo Structure

```
Repo Root
|---- arch                  # (FATE internal)
|---- cluster-deploy        # (FATE internal)
|---- examples              # Example datasets and configurations of FLASHE
|---- deployment            # Procedures for deploying and evaluating FLASHE
|---- fate_flow             # (FATE internal)
|---- federatedml           # (FATE internal)
|---- simulation            # Procedures related to simulation in FLASHE
|---- utils                 # Tools helpful for playing with FATE/FLASHE
|---- eggroll               # (FATE generated)
|---- fate-flow             # (FATE generated)
```

# Notes
Please consider to cite our paper if you use the code or data in your research project.
```bibtex
@inproceedings{FLASHE-arxiv21,
  title={FLASHE: Additively Symmetric Homomorphic Encryption for Cross-Silo Federated Learning},
  author={Zhifeng Jiang and Wei Wang and Yang Liu},
  booktitle={arXiv:2109.00675},
  year={2021}
}
```

# Acknowledgement
Thanks to [Chengliang Zhang](https://github.com/marcoszh) and [Junzhe Xia](#) for their engineering advice regarding FATE. Thanks to [Minchen Yu](https://github.com/MincYu) for his suggestions on development automation.

# Contact
Zhifeng Jiang (zjiangaj@cse.ust.hk)
