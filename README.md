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
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = [10, 7.5]

ar1 = np.array([1,0.33])
ma1 = np.array([1,0.9])
ARMA_1 = ArmaProcess(ar1,ma1).generate_sample(nsample = 1000)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_1)
plot_pacf(ARMA_1)
ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=10000)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_2)
plot_pacf(ARMA_2)
```

### OUTPUT:
### NAME : DEVADHAARINI.D
### REGNO: 212223230040
SIMULATED ARMA(1,1) PROCESS:

<img width="846" height="643" alt="image" src="https://github.com/user-attachments/assets/32774bfb-fc9f-44f9-a5be-6dbf152f9ecd" />


Partial Autocorrelation

<img width="851" height="649" alt="image" src="https://github.com/user-attachments/assets/a75de652-1b2e-4e10-b61f-a628de5b89f8" />

Autocorrelation

<img width="873" height="649" alt="image" src="https://github.com/user-attachments/assets/cc3190f7-41fd-46db-b49e-78471f923a0b" />

SIMULATED ARMA(2,2) PROCESS:

<img width="866" height="645" alt="image" src="https://github.com/user-attachments/assets/e1a4741c-3720-4596-bfcd-c4dd5f3dfac4" />

Partial Autocorrelation

<img width="880" height="643" alt="image" src="https://github.com/user-attachments/assets/ff0ecccd-a0ce-464c-ab54-585c94b55b33" />

Autocorrelation
<img width="852" height="642" alt="image" src="https://github.com/user-attachments/assets/45f76eaf-69de-4e72-93ab-904518772a50" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
