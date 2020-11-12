import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sns.set()

raw = pd.read_csv('2.01. Admittance.csv')
#print(raw.head(), raw.describe())

mapped = raw.copy()
mapped['Admitted'] = mapped['Admitted'].map({'Yes':1,'No':0})
#print(mapped.head())

y = mapped['Admitted']
x1 = mapped['SAT']
plt.scatter(x1, y, color = 'C0')
plt.xlabel('SAT')
plt.ylabel('Admitted')
#plt.show()
plt.close('all')

# PLOTTING A LINEAR REGRESSION LINE
x = sm.add_constant(x1)
reg_lin = sm.OLS(y,x)
results_lin = reg_lin.fit()

plt.scatter(x1,y, color = 'C0')
yhat = x1*results_lin.params[1]+results_lin.params[0]

plt.plot(x1,yhat,lw=2.5,color='C8')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('Admitted', fontsize = 20)
#plt.show()
plt.close('all')


# PLOTTING A LOGISTIC REGRESSION
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()
# creating a logit function
def f(x,b0,b1):
    return np.array(np.exp(b0+x*b1) / (1 + np.exp(b0+x*b1)))
# sorting x and y so we can plot the curve
f_sorted = np.sort(f(x1, results_log.params[0], results_log.params[1]))
x_sorted = np.sort(np.array(x1))
plt.scatter(x1, y, color = 'C0', alpha = 0.2)
plt.xlabel('SAT')
plt.ylabel('Admitted')
plt.plot(x_sorted, f_sorted, color = 'C8')
plt.show()
