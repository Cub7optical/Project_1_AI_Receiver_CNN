# Project_1_AI_Receiver_CNN

Code Description:
The following code was created for the purpouse of evaluating the functionality of a convolutional neural network (CNN) model.
My data is sourced from a four chanel wave division multiplexed (WDM) signal where the ground truth TX signal has been left in tact and the RX data has been distorted and effected non-linearly. For simplicity I will use examples that relate to the dispersion data. To learn the dataset and unsort the dispersion I used a three level CNN and invorporated a sigmoid activation function to discern between 0 and 1.

Data description:
file name:
TX_lambda1.csv
RX_lambda1.csv
TX_lambda2.csv
RX_lambda2.csv
TX_lambda3.csv
RX_lambda3.csv
TX_lambda4.csv
RX_lambda4.csv

Key atributes:
  Sample rate in samples-per-second = 400e9  
  Symbol rate in symbols-per-second = 25e9 
  Number of symbols                 = 16384  
  Symbol period                     = 1 / 25e9
  Samples-per-symbol                = 16

Function Description:
read_csv_data_iterative.py: This function reads in the WDM data indavidually for each .csv file
create_model.py: This function establishes the model throughout the code. It has been duplicated twice to support the receptive field and decimation evaluations
