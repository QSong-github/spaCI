The data structure for spaCI training is shown as below:
```
|spaCI
├── dataset
│     ├── exp_data_LR.csv
│     ├── triplet.csv
│     ├── test_pairs.csv
│     ├── test_lr_pairs.csv
│     ├── spatial_graph.csv
```

* exp_data_LR.csv: spatial gene expression of ligands and receptors.
* triplet.csv: gene triplet, please refer to our paper for details.
* validation_pairs.csv: gene pairs used for spaCI validation, which consists of one column of known labels.
* test_lr_pairs.csv: the list of ligand-receptor pairs for prediction. This dataset does not need labels, or you can use a constant label instead.
* spatial_graph.csv: spatial cell adjacency graph, in the form of adjacent list.


To generate the above data list, you only need to prepare two csv files please refer to [here](https://github.com/QSong-github/spaCI/tree/main/dataset/example_data) for our toy data example):
* st_expression.csv: spatial cell gene expressions, where rows refer to genes and columns are cells.
* st_meta.csv: the columns stand for the dimx, dimy, cell type. Rownames refer to cells. Dimx and dimy are the spatial locations of cells.


With these two csv file, you can generate the spaCI needed data structure with one R script command:

```
cd src
/path/to/Rscript spaCI_preprocess.R /path/to/st_expression.csv /path/to/st_meta.csv K p /path/to/database /path/to/saved_dir
```

We prepared to hyperparameters for generate the lr_paris and graph, where     
* K: the number of spatial located neighbors for each cell     
* p: cutoff of selecting top associated gene pairs
* Database: L-R interaction database
* saved_dir: The folder path of saved data

For example, you may run the following command line with specified parameters:
```
cd src
Rscript spaCI_preprocess.R ../dataset/example_data/st_expression.csv ../dataset/example_data/st_meta.csv  5  0.01 'spaCI_database.RDS' ../dataset/real_data
```

To run the above preprocess step, you may need to install two R packages:
```
install.packages("Matrix")
```
```
install.packages("RANN") or devtools::install_github("jefferis/RANN")
```
