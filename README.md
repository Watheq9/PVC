# PVC:  A Podcast Test Collection with Transcript and Query Variations


## Libraries Installation
Before replicating the work, you need to create a conda environment, activate it, and then install the needed libraries. Run the following commands:

```
conda create --name PVC python=3.10
conda activate PVC
conda install -c pytorch faiss-cpu pytorch -y
pip install -r requirements.txt
```

If you do not already have JDK 21 installed, install via conda:
```
conda install -c conda-forge openjdk=21 maven -y
```


1. To generate the transcription, refer to transcription folder


2. To build the pool, refer to the systems dir
