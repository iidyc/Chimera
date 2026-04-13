wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar -xvzf collectionandqueries.tar.gz
rm collectionandqueries.tar.gz
awk -F'\t' '{OFS="\t"; $1=NR; print}' queries.dev.small.tsv > tmp && mv tmp queries.dev.small.tsv