# Good. This is exactly how you should study — map questions → known code patterns.

I’ll give you ONLY CODE + SHORT COMMENTS, strictly aligned with your previous notebooks.


---

🔥 LOGISTICS PAPER (Probability) 

✅ Task 1 — Bernoulli

import numpy as np
import matplotlib.pyplot as plt

p = 0.85

X = np.random.choice([0,1], size=500, p=[1-p, p])   # simulate trials

print(np.mean(X))        # observed mean
print(np.var(X))         # observed variance

print(p)                 # theoretical mean
print(p*(1-p))           # theoretical variance

plt.hist(X, bins=2)      # visualize
plt.show()


---

✅ Task 2 — Binomial

from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

n = 20
p = 0.9

x = np.arange(0, n+1)
pmf = binom.pmf(x, n, p)       # probabilities

print(binom.pmf(18, n, p))     # P(X=18)

plt.bar(x, pmf)                # plot
plt.show()

print(x[np.argmax(pmf)])       # mode


---

✅ Task 3 — Poisson

from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt

lam = 4

x = np.arange(0, 15)
pmf = poisson.pmf(x, lam)      # probabilities

print(poisson.pmf(4, lam))     # P(X=4)
print(poisson.pmf(0, lam))     # P(X=0)

plt.bar(x, pmf)
plt.show()


---

✅ Task 4 — Geometric

from scipy.stats import geom
import numpy as np
import matplotlib.pyplot as plt

p = 0.1

x = np.arange(1, 16)
pmf = geom.pmf(x, p)           # probabilities

print(geom.pmf(5, p))          # first failure at 5

plt.bar(x, pmf)
plt.show()


---

✅ Task 5 — Poisson Approximation

from scipy.stats import poisson

n = 2000
p = 0.002

lam = n * p                    # poisson parameter

print(lam)

print(1 - poisson.cdf(2, lam)) # P(X>2)


---

🔥 LLN 

import numpy as np
import matplotlib.pyplot as plt

n = 2000

data = np.random.binomial(1, 0.5, n)      # coin flips

avg = np.cumsum(data) / np.arange(1, n+1) # running mean

plt.plot(avg)
plt.axhline(y=0.5)                        # expected value
plt.show()


---

🔥 DATA PIPELINE 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(10)

# Data generation
Age = np.random.randint(18, 70, 150)
noise = np.random.normal(0, 200, 150)
Spending = Age*50 + noise

df = pd.DataFrame({'Age': Age, 'Spending': Spending})

# Missing values
df.loc[np.random.choice(df.index, 10), 'Spending'] = np.nan
df['Spending'].fillna(df['Spending'].mean(), inplace=True)

# Categorical
df['Member'] = np.random.choice(['Gold','Silver','Bronze'], 150)

# Binary column
df['High'] = (df['Spending'] > 2000).astype(int)

# Plots
sns.boxplot(x='Member', y='Spending', data=df)
plt.show()

plt.scatter(df['Age'], df['Spending'])
plt.show()


---

🔥 CLT 

import numpy as np
import matplotlib.pyplot as plt

def dice_sum(n_dice, n_samples=1000):
    return [np.sum(np.random.randint(1,7,n_dice)) for _ in range(n_samples)]

sum10 = dice_sum(10)
sum50 = dice_sum(50)

plt.hist(sum10, bins=30)    # less normal
plt.show()

plt.hist(sum50, bins=30)    # more normal
plt.show()


---

🔥 PCA1 (Wine) 

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

wine = load_wine()
X = wine.data[:, :6]
y = wine.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)   # variance

plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.show()


---

🔥 PCA2 (Breast Cancer) 

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(X.shape, X_pca.shape)             # shape change
print(pca.explained_variance_ratio_)   # variance

plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.show()


---

🔥 PCA3 (Penguins) 

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = sns.load_dataset('penguins')

df = df.dropna()   # remove missing

X = df[['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']]
y = df['species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)   # variance

plt.scatter(X_pca[:,0], X_pca[:,1], c=y.astype('category').cat.codes)
plt.show()


---
