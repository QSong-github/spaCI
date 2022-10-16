
args <- commandArgs(trailingOnly = TRUE)
print(args)

st.file = args[[1]]
meta.file  = args[[2]]
k=as.numeric(args[[3]])
cut=as.numeric(args[[4]])
dir0=args[[5]]

st.exp <- read.csv(file=st.file,stringsAsFactors=F,check.names=F,row.names=1)
st.meta <- read.csv(file=meta.file,stringsAsFactors=F,check.names=F,row.names=1)                    
source('spaCI_utils.R')
LRdB <- readRDS('L-R database.RDS')

spaCI_preprocess(st.exp = st.exp,
                 st.meta = st.meta,
                 LRdb = LRdB,
                 K=k,
                 cutoff=cut,
                 dir=dir0,
                 cor_eval='pearson')


