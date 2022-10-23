We use Rscripts to generate the data for training spaCI:
```
|spaCI
├── dataset
│     ├── exp_data_LR.csv
│     ├── triplet.csv
│     ├── test_pairs.csv
│     ├── test_lr_pairs.csv
│     ├── spatial_graph.csv
```

* exp_data_LR.csv: 
* triplet.csv: 
* test_paris.csv:
* test_lr_paris.csv: 
* spatial_graph.csv: 


You need to prepare the two csv files please refer to [here](https://github.com/tonyyang1995/spaCI/tree/main/dataset/example_data) for our toy data example):
* st_expression.csv: rows is cells, columns genes
* st_meta.csv: rows: x,y,cell_types, 


With the two csv file, you can generate the lr pairs with the R script command:
We prepared to hyperparameters for generate the lr_paris and graph, where     
* K: the hyper-parameters for KNN     
* p: the hypte-parameters for cutoff.
* Database: 
* save_dir:   
```
cd src
/path/to/Rscript spaCI_preprocess.R /path/to/st_expression.csv /path/to/st_meta.csv k p /path/to/database.RDS /path/to/saved/dir
```

For example, in our local settings, we are using the following command line:
```
cd src
Rscript spaCI_preprocess.R ../dataset/exaample_data/st_expression.csv ../dataset/exaample_data/st_meta.csv 5 0.01 'spaCi_database.RDS' ../dataset/real_data
```

You may need to install the following R packages:
```
install.packages("Matrix")
install.packages("RANN")
```
sometimes, RANN cannot be installed from install.packages, you can try an alternative method from github:
```
install.packages("devtools")
devtools::install_github("jefferis/RANN")

```
