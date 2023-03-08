# PSPred-AEF

1 Description

Plant vacuoles are the most important organelles for plant growth, development, and defense, and they play an important role in many types of stress responses. An important function of vacuole proteins is the transport of various classes of amino acids, ions, sugars, and other molecules. Accurate identification of plant vacuole proteins (PVPs) is crucial for revealing their biological functions. Here, we developed a new machine learning-based software that enables the classification of proteins into PVPs or non-PVPs specifically and effectively. This predictive software was designed using a light gradient boosting machine classifier and hybrid features composed of deep representation learning features (UniRep and BiLSTM) and classic sequence features (DDE and DPC). It has the potential to facilitate future computational work in this field. Webserver and datasets are also available at: http://lab.malab.cn/~acy/iPVP-DRLF. Based on sequence length analysis of three datasets (training, testing and blind), we recommend users use iPVP-DRLF for prediction if their protein sequence length is between 50 and 2000.

2 Requirements

Before running, please make sure the following packages are installed in Python environment:

python==3.8

pytorch==1.13.1

numpy==1.24.2

pandas==1.5.3



3 Running

Download the BiLSTM embedding model here, and put it in the directory named "Model";

Changing working dir to iPVP-DRLF-main, and then running the following command:

python iPVP-DRLF-main.py -i test.fasta -o prediction_results.csv

-i: input file in fasta format

-o: output file name
