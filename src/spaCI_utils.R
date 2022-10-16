options(warn=-1)

#'# Class definitions
#' @importFrom methods setClassUnion
#' @importClassesFrom Matrix dgCMatrix
library(Matrix)
setClassUnion(name = 'AnyMatrix', members = c("matrix", "dgCMatrix","data.frame"))
setClassUnion(name = 'AnyFactor', members = c("factor", "list",'character'))

my.sapply <- ifelse(
    test = future::nbrOfWorkers() == 1,
    yes = pbapply::pbsapply,
    no = future.apply::future_sapply)

#' The key slots used in the spaCI object are described below.
#' @slot data.raw raw count data matrix
#' @slot data.normalize normalized data matrix 
#' @slot data.scale scaled data matrix
#' @slot data.project name of data
#' @slot meta data frame storing the information associated with each cell
#' @slot idents a factor defining the cell identity
#' @exportClass spaCI
#' @importFrom methods setClass
spaCI <- methods::setClass("spaCI",
                           slots = c(data.raw = 'AnyMatrix',
                                     data.normalize = 'AnyMatrix',
                                     data.project = "AnyFactor",
                                     LR = 'AnyMatrix',
                                     cor = 'AnyMatrix',
                                     spaCI_output='AnyMatrix',
                                     cci = "list",
                                     meta = "data.frame",
                                     idents = "AnyFactor"
                                     )
                           )

#' This function normalize count data
normalize_data <- function(count){
    #require(Seurat)
    norm <- as.matrix(Seurat:::NormalizeData.default(count,verbose=F))
  return (norm)}

#' @param database: LR database
#' @param object: spaCI object
#' @export prepared data for spaCI
lr_triplet <- function(object, database, threshold, cuttoff, method){
    cts <- object@data.raw
    
    ps <- which(rowSums(cts == 0) < threshold)
    cts1 <- cts[ps,]

    if (method == 'pearson'){
        cor.res <- cor(t(cts1),method='pearson')
    } else if (method=='spearman'){
        cor.res <- cor(t(cts1),method='spearman')
    } else if (method=='kendall'){
        cor.res <- cor(t(cts1),method='kendall')
    }
    
    diag(cor.res) <- 0 
    temp <- as.numeric(cor.res)
    temp1 <- temp[order(temp,decreasing=T)]
    
    cutoff.h <- temp1[round(length(temp1)*cuttoff)]
    
    if (cutoff.h >0.5){
        #' generate triplet with pos & neg
        pos.ps <- which(cor.res > cutoff.h, arr.ind=T)
        all_genes <- rownames(cts1)
        pos.lrs <- cbind(all_genes[pos.ps[,1]],all_genes[pos.ps[,2]])
        pos.lrs <- data.frame(pos.lrs)
        sel = round(nrow(pos.lrs)/2)
        
        trip <- sapply(1:sel,function(t){
            tem <- cbind(pos.lrs[t,],colnames(cor.res)[order(cor.res[pos.lrs[t,1],])][1:3])
            colnames(tem) <- c('key','pos','neg')
            return (tem)
            })
        
        triplet <- data.frame(do.call(rbind,trip))
        colnames(triplet) <- c('key','pos','neg')
        
        #' generate test pairs with pos & neg
        foo <- unique(abs(as.numeric(cor.res)))
        foo1 <- foo[order(foo,decreasing=F)]
        cutoff.l <- foo1[round(length(foo1)*cuttoff)]
        #if (cutoff.l==0){cutoff.l=0.1}
        neg.ps <- which((abs(cor.res) <= cutoff.l) & (abs(cor.res) >0 ),arr.ind=T)
        neg.lrs <- cbind(all_genes[neg.ps[,1]],all_genes[neg.ps[,2]])
        test.p <- pos.lrs[(sel+1):nrow(pos.lrs),]
        test.p$label <- 1
        test.n <- data.frame(neg.lrs)
        test.n$label <- 0
        test_pairs <- rbind(test.p, test.n)
        colnames(test_pairs)[1:2] <- c('ligand','receptor') 

        index <- unlist(my.sapply(1:nrow(database),function(i){
            l1 <- database[i,1]; l2 <- database[i,2]
            t1 <- rownames(cts)==l1; t2 <- rownames(cts)==l2
            if ( sum(t1)+sum(t2)==2){ return (i) }}))

        test.lrs <- database[index,1:2]
    
        return (list(triplet,test_pairs,test.lrs))
    } else {
        print ('no strong association detected')
    }
}


#' @param st.exp raw spatial data matrix
#' @param st.meta meta data with dimx, dimy, and cell type (optional)
#' @param LRdb L-R pair database
#' @param K cell neighbors in spatial data
#' @param cutoff top L-R pairs selected for training spaCI model
#' @param cor_eval correlation method for gene-gene associations 
#' @export spaCI_preprocess
spaCI_preprocess <- function(st.exp, st.meta, LRdb, K, cutoff,
                             dir, cor_eval='pearson'){

    dir1 = dir
    lrs <- unique(c(LRdb[,1],LRdb[,2]))
    norm.exp <- normalize_data(st.exp)
    #' @param spaci.obj object 
    spaci.obj <- methods::new(Class = "spaCI",
                              data.raw = st.exp,
                              data.normalize = norm.exp,
                              data.project = 'spaCI',
                              meta = st.meta,
                              idents = st.meta$type)

    N1 <- ncol(st.exp)
    #' select training set 
    gt.list <- lr_triplet(object = spaci.obj,
                          database=LRdb,
                          threshold = N1/2,
                          cuttoff=cutoff, method=cor_eval)
    #' test set
    test.lrs =  data.frame(gt.list[[3]])
    test.lrs$trueLabel = 0
    
    #' save triplets
    #dir1 <- './data_IO'; dir.create(dir1)
    write.csv(gt.list[[1]],file=paste0(dir1,'/triplet.csv'),quote=F)
    write.csv(gt.list[[2]],file=paste0(dir1,'/test_pairs.csv'),quote=F)
    write.csv(test.lrs,file=paste0(dir1,'/test_lr_pairs.csv'),quote=F)
    write.csv(st.exp,file=paste0(dir1,'/exp_data_LR.csv'),quote=F)

    #' generate adjacent graph
    library(RANN); df = st.meta[,c('x','y')]
    closest <- RANN::nn2(data = df, k = K)[[1]]
    
    adj <- matrix(0, nrow=N1, ncol=N1)
    for (i in 1:N1){
        adj[i,closest[i,1:K]] = 1
        adj[closest[i,1:K],i] = 1
    }
    rownames(adj)=colnames(st.exp); colnames(adj)=colnames(st.exp)
    write.csv(adj,file=paste0(dir1,'/spatial_graph.csv'),quote=F)
}
