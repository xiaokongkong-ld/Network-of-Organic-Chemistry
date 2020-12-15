# Network-of-Organic-Chemistry
1. data:
raw data from https://figshare.com/articles/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873

“fingerPrintIndexedLinks.txt” edges lists extracted, used as dataset in experiments

“compoundsWithFingerprintMatrix.txt” nodes lists with molecular fingerprint (features) extracted, used as dataset in experiments

2. extract-reactions:
Java programs to extract reactions and build datasets

"findFingerPrint.java" ,"batchReader"to extract reaction SMILES of each reactions

"fingerprintTxtReader.java" to separate reactant and product in a reaction

"fingerPrintIndex.java" to index every compounds (including reactants and products) by number.

"set_finger_print.py" to add bit vector style fingerprint to every compound.

"helper.java" to help format the datasets’s fingerprints

3. Collaborative Filtering:
Collaborative filtering experiments. 

4. matrix_factorization_grad_descent.py:
Matrix factorization experiment

5. draw_network.py:
To visuallize NOC networks and calculate network statistics.

6. link_prediction_machine_learning.py:
Link prediction experiment by machine learning

7. LinkPrediction gcn.py:
Link prediction by GCN, use tf-geometric library
