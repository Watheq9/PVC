



1. Download the ColBERT v2 model:

```wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz```


2. Setupt the conda environment:

```
conda create --name colbert python=3.8
source activate colbert

git clone https://github.com/stanford-futuredata/ColBERT.git
pip install -e ColBERT/['faiss-gpu','torch']
conda install -c conda-forge cudatoolkit-dev -y 

```




3. Build the indexes using ``run_indexing.sh`` script. You need to set the path to the downloaded model.


4. Run the retrieval using ``run_retrieval.sh``