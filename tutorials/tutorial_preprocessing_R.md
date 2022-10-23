
  We provided an R script to help you generate the triplets and lr pairs. All you need is two csv files: (st_expression.csv and st_meta.csv, please refer to [here](https://github.com/tonyyang1995/spaCI/tree/main/dataset/example_data) for our toy data example). 
  The st_expression.csv is a 2D matrix, the columns contains the receptors, and rows are the ligands.
  The st_meta.csv is the meta files, the columns contains the x,y and cell_type, and rows are the information of ligands.

With the two csv file, you can generate the lr pairs with the R script command:
We prepared to hyperparameters for generate the lr_paris and graph, where     
  K is the hyper-parameters for KNN     
  p is the hypte-parameters for cutoff.
```
cd src
/path/to/Rscript spaCI_preprocess.R /path/to/st_expression.csv /path/to/st_meta.csv 5 0.02 /path/to/saved/dir
```