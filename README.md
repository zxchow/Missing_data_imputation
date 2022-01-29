# Missing_data_imputation

This repository implements six different data imputation approaches in tensorflow 2.7.0 and compares the performances in imputating 5-minutes HSI data. 

The following approaches are included:
1. MRNN.py: "Estimating Missing Data in Temporal Data Streams Using Multi-directional Recurrent Neural Networks"
2. GRU-D.py: "Recurrent Neural Networks for Multivariate Time Series with Missing Values"
3. NAOMI.py: "NAOMI: Non-Autoregressive Multiresolution Sequence Imputation"
4. En-decoder.py: "Learning from Irregularly-Sampled Time Series: A Miss Data Perspective"
5. DeepMVI.py: "Missing Value Imputation on Multidimensional Time Series"
6. SSIM.py: "SSIM—A Deep Learning Approach for Recovering Missing Time Series Sensor Data"

We use the past five years 5-minute HSI data, including its open, high, low, close and volume (5 channels in total), as the basic dataset in this experiment. To do the training and testing, we separate the sequence of HSI data into around 1000 data clips with a length of 100. 70% of these data clips are randomly selected for training while the rest are used for testing. 

the MSE loss is used as the criteria for comparing their performances in the experiment. https://www.tensorflow.org/api_docs/python/tf/keras/metrics/mean_squared_error. In addition, we record the average running time of each approach for each epoch as a reference.

In the experiment, We have two types of imputation mode: low missing rate (20%) and high missing rate (40%). For the mode of low missing rate, we mask 20% of the raw data in the center of each data clip and use the rest to imputate them. For the mode of high missing rate, everything is the same as the low one except 40% data are masked in the center. 

To reproduce our results, please run the main.py

# Results of low missing rate


| Approach  | MSE loss (the lower the better) |  Running time (second) |
| ------------- | ------------- | ---------- |
| MRNN  | 7590.74  | 2.36  |
| GRU-D  | 5690.44  | 3.99  |
| NAOMI  | 5684.99  | 14.48  |
| En-decoder  | 299.37 | 1.54  |
| DeepMVI  | 326.89  | 0.58  |
| SSIM  | 26796.87  | 23.24  |

In mode of low missing rate, En-decoder reaches the best imputation accuracy of a 299.37 mse loss with a average running time of 1.54 seconds. Compared to En-decoder, DeepMVI attains a slightly higher imputation error of a 326.89 mse loss but it shortens the running time significatly, from 1.54 seconds to 0.58 second. Other methods perform worse than the aforementioned two approaches in terms of both imputation accuracy and running time.

In a word, we should use En-decoder for data imputation as it has the lowest imputation error. However, in some speical cases where time-performance is important, such as high-frequency trading, DeepMVI could be a better choice as it can run significantly faster with a slightly higher mse loss.

# Results of high missing rate


| Approach  | MSE loss (the lower the better) |  Runing time |
| ------------- | ------------- | -----------|
| MRNN  | 7591.04  | 2.25  |
| GRU-D  | 5679.79  | 3.98  |
| NAOMI  | 5800.70  | 17.82  |
| En-decoder  | 346.30  | 1.87  |
| DeepMVI  | 417.65  | 0.62  |
| SSIM  | 37479.36  | Content Cell  |

In the mode of high missing rate, En-decoder also attains a lowest MSE loss of only 346.30 followed by the DeepMVI with a mse loss of 417.65. Compared the results in low missing rate, the distance between DeepMVI and En-decoder is increased. This distance could become enen larger along with the rising of the missing rate. Therefore, despite DeepMVI requires a shorter running time, it would be better to choose En-decoder to maintain an accurate imputation if the missing rate is high.
