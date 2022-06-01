# DIPOLE (DIstributed Persistence-Optimized Local Embeddings)
DIPOLE is a dimensionalityreduction post-processing step that corrects an initial embedding by minimizing a loss functional with both a local, metric term and a global, topological term. By ﬁxing an initial embedding method (we use Isomap), DIPOLE can also be viewed as a full dimensionality-reduction pipeline. This framework is based on the strong theoretical and computational properties of distributed persistent homology and comes with the guarantee of almost sure convergence. We observe that DIPOLE outperforms popular methods like UMAP, t-SNE, and Isomap on a number of popular datasets, both visually and in terms of precise quantitative metrics.

The original version and code of DIPOLE can be found here: https://github.com/aywagner/DIPOLE. In this version we have made changes to DIPOLE so that it is easier to select on which dataset we want to test it on. 


## Start up 
### Requirements
This codebase is written entirely in Python 3.7. The file requirements.txt contains all necessary packages to run the code successfully. These are easy to install via pip using the following instruction:
```bash
pip install -r requirements.txt
```
Or using conda:
```bash
conda install --file requirements.txt
```

### In Action
By calling
```bash
python runme.py
```

you are provided with a command line interface. The instructions are straight forward.


## Datasets
brain1.npy is derived from data made available at https://www.insight-journal.org/midas/community/view/21 by Bullitt, Elizabeth; Smith, J Keith; Lin, Weili.

Mammoth data downloaded from https://github.com/PAIR-code/understanding-umap and originally from https://3d.si.edu/object/3d/mammuthus-primigenius-blumbach:341c96cd-f967-4540-8ed1-d3fc56d31f12.

Stanford faces dataset face_data.mat downloaded from http://isomap.stanford.edu/datasets.html.
