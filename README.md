# Data

No data files are committed to this repository.

## CPTAC proteogenomic data

Install the `cptac` Python package and download datasets on first use:

```python
pip install cptac

import cptac
ds = cptac.Gbm()          # replace with tumor of interest
prot  = ds.get_proteomics(tissue_type="tumor")
trans = ds.get_transcriptomics(tissue_type="tumor")
cnv   = ds.get_CNV(tissue_type="tumor")
clin  = ds.get_clinical()
```

Available tumor types: `Gbm`, `Luad`, `Lscc`, `Ucec`, `Ccrcc`,
`Hnscc`, `Pdac`, `Ovarian`, `Brca`, `Coad`.

Full documentation: https://github.com/PayneLab/cptac

## TCGA gene expression data

TCGA PanCancer Atlas mRNA z-scores are available via cBioPortal:
https://www.cbioportal.org

Survival data and expression profiles can be retrieved programmatically
via the cBioPortal REST API (https://www.cbioportal.org/api).
