# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data=pd.read_csv('AirPassengers.csv')
N=1000
plt.rcParams['figure.figsize'] = [12, 6] #plt.rcParams is a dictionary-like object in Mat
X=data['#Passengers']
plt.plot(X)
plt.title('Original Data')
plt.show()
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')

plt.xlim([0, 500])
plt.show()
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```

### OUTPUT:
### NAME : DEVADHAARINI.D
### REGNO: 212223230040
SIMULATED ARMA(1,1) PROCESS:

<img width="976" height="516" alt="image" src="https://github.com/user-attachments/assets/e3d3ae0f-37a7-46c5-bc34-c6a3fd559c44" />

Partial Autocorrelation

<img width="984" height="514" alt="image" src="https://github.com/user-attachments/assets/14ac8b47-8706-4cea-b5c2-76b39a676466" />

Autocorrelation

<img width="988" height="519" alt="image" src="https://github.com/user-attachments/assets/031870d8-c622-4ab5-90b3-16a1ee782bef" />

SIMULATED ARMA(2,2) PROCESS:

<img width="975" height="523" alt="image" src="https://github.com/user-attachments/assets/b87ac4a8-b5a7-4ab3-9138-47966902082b" />

Partial Autocorrelation

<img width="982" height="514" alt="image" src="https://github.com/user-attachments/assets/5ad22978-7334-461d-be30-869dcf29aadd" />

Autocorrelation

<img width="993" height="513" alt="image" src="https://github.com/user-attachments/assets/f7092247-ee1f-4e08-b503-0f898f7b5367" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
