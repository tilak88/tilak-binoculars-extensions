# Project results

The archived run (notebooks/bino_extension.ipynb) reports a balanced 5,422-sample evaluation across CNN-Falcon, CNN-Llama, PubMed, and HC3 using a 256-token cap and a single joint observer/performer pass.

| Feature | Average/result reported | Interpretation |
|---|---:|---|
| Binoculars baseline | AUROC 0.976 average | Strongest standalone reference |
| STC | AUROC 0.940 average | Strong retained novel feature |
| CMRD-var | AUROC 0.709 average | Most orthogonal feature; largest novel fusion contribution |
| Full HC3 fusion | 0.9873 → 0.9923 AUROC | +0.005 cumulative gain in the reported ablation |

The results are benchmark-specific. The repository does not claim universal detector accuracy, language independence, or deployment readiness.

