# Missing_data_imputation

This repository implement six different data imputation approaches and compares the performances in imputating 5-minutes HSI data. 

The following approachs are included:
1. MRNN.py: "Estimating Missing Data in Temporal Data Streams Using Multi-directional Recurrent Neural Networks"
2. GRU-D.py: "Recurrent Neural Networks for Multivariate Time Series with Missing Values"
3. NAOMI.py: "NAOMI: Non-Autoregressive Multiresolution Sequence Imputation"
4. En-decoder.py: "Learning from Irregularly-Sampled Time Series: A Miss Data Perspective"
5. DeepMVI.py: "Missing Value Imputation on Multidimensional Time Series"
6. SSIM.py: "SSIM—A Deep Learning Approach for Recovering Missing Time Series Sensor Data"

We use the past-five years 5-minute HSI data, including its open, high, low, close and volume (5 channels in total), as the basic dataset in this experiment. To do the training and testing, we separate the sequence of HSI data into arond 1000 data clips with a length of 100. 70% of these data clips are randomly selected for training which the rest are used for testing. 

the MSE loss is used as the criteria for comparing their performances in the experiment. https://www.tensorflow.org/api_docs/python/tf/keras/metrics/mean_squared_error. In addition, we also record the average running time of each approach for each epoch as a reference.

In the experiment, We have two types of imputation mode: small missing (20% miss) and large missing (40% miss). For the mode of small missing, we mask the 20% data in the center of each data clip and use the resting to imputate them. For the mode of large miss, everything is the same with the samll one except 40% data are masked in the center. 

To reproduce our results, please run the main.py

# Small missing results

| Approach  | MSE loss (the lower the better) |  Running time (second) |
| ------------- | ------------- |
| MRNN  | 7590.74  | 2.36  |
| GRU-D  | 26664.39  | 3.77  |
| NAOMI  | 5684.99  | 14.48  |
| En-decoder  | 299.37 | 1.54  |
| DeepMVI  | 326.89  | 0.58  |
| SSIM  | 26796.87  | 23.24  |

In small missing mode, En-decoder reaches the best imputation accuracy of a 299.37 mse loss with a average running time of 1.54 second. Compared to En-decoder, DeepMVI attains a slightly higher imputation error of a 326.89 mse loss but it shortens the running time significatly, from 1.54 second to 0.58 second. Other methods perform worse than the aforementioned two approaches in terms of both imputation accuracy and running time.

In a word, we should use En-decoder for data imputation as it has the lowest imputation error. However, in some speical cases where time-performance is important, such as high-frequency trading, DeepMVI could be a better choice as it can run significantly faster with a slightly higher mse loss.

# Large miss results


| Approach  | MSE loss (the lower the better) |  Runing time |
| ------------- | ------------- |
| MRNN  | Content Cell  | Content Cell  |
| GRU-D  | Content Cell  | Content Cell  |
| NAOMI  | Content Cell  | Content Cell  |
| en-decoder  | Content Cell  | Content Cell  |
| DeepMVI  | Content Cell  | Content Cell  |
| SSIM  | Content Cell  | Content Cell  |
