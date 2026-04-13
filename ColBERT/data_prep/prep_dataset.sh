# conda install conda-forge::colbert-ai
mkdir -p msmarco
mkdir -p hotpot
cd msmarco
bash download_msmarco.sh
cd ../hotpot
bash download_hotpot.sh
cd ..

python ColBERT/encode_dataset.py msmarco/collection.tsv msmarco/msmarco_emb.bin msmarco/msmarco_doclens.bin
python ColBERT/encode_dataset.py hotpot/corpus.tsv hotpot/hotpot_emb.bin hotpot/hotpot_doclens.bin
python ColBERT/encode_queries.py msmarco/queries.dev.small.tsv msmarco/msmarco_queries.bin
python ColBERT/encode_queries.py hotpot/queries.tsv hotpot/hotpot_queries.bin
python compute_topk.py msmarco/msmarco_emb.bin msmarco/msmarco_doclens.bin msmarco/msmarco_queries.bin msmarco/msmarco_gt.tsv
python compute_topk.py hotpot/hotpot_emb.bin hotpot/hotpot_doclens.bin hotpot/hotpot_queries.bin hotpot/hotpot_gt.tsv
