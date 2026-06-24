import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('bike_sharing.csv')

print(df.head())
print(df.info())

# =========================
# DATE PROCESSING
# =========================
df['dteday'] = pd.to_datetime(df['dteday'])
df['day'] = df['dteday'].dt.day
df['month'] = df['dteday'].dt.month
df['year'] = df['dteday'].dt.year

# =========================
# DROP USELESS / LEAKAGE
# =========================
df = df.drop(['instant','casual','registered'], axis=1)

# =========================
# MAPPING (LIKE VIDEO)
# =========================
df['weekday'] = df['weekday'].map({
    0:'Sun',1:'Mon',2:'Tue',3:'Wed',
    4:'Thu',5:'Fri',6:'Sat'
})

df['month'] = df['month'].map({
    1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
    7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'
})

df['season'] = df['season'].map({
    1:'spring',2:'summer',3:'fall',4:'winter'
})

df['weathersit'] = df['weathersit'].map({
    1:'A',2:'B',3:'C',4:'D'
})

# =========================
#  VALUE COUNTS 
# =========================
print("\nCategory Counts:\n")
print(df['season'].astype('category').value_counts())
print(df['month'].astype('category').value_counts())
print(df['weekday'].astype('category').value_counts())
print(df['workingday'].astype('category').value_counts())

# =========================
#  SUBPLOTS
# =========================
plt.figure(figsize=(18,10))

plt.subplot(2,3,1)
sns.boxplot(x='season', y='cnt', data=df)

plt.subplot(2,3,2)
sns.boxplot(x='month', y='cnt', data=df)

plt.subplot(2,3,3)
sns.boxplot(x='weekday', y='cnt', data=df)

plt.subplot(2,3,4)
sns.boxplot(x='weathersit', y='cnt', data=df)

plt.subplot(2,3,5)
sns.boxplot(x='workingday', y='cnt', data=df)

plt.tight_layout()
plt.show()

# =========================
# DISTRIBUTIONS
# =========================
sns.histplot(df['temp'], kde=True)
plt.title("Temp Distribution")
plt.show()

sns.histplot(df['atemp'], kde=True)
plt.title("Atemp Distribution")
plt.show()

sns.histplot(df['hum'], kde=True)
plt.title("Humidity Distribution")
plt.show()

sns.histplot(df['windspeed'], kde=True)
plt.title("Windspeed Distribution")
plt.show()

# =========================
# PAIRPLOT
# =========================
numeric_df = df.select_dtypes(include=['int64','float64'])

sns.pairplot(numeric_df)
plt.show()

# =========================
# TRIANGLE HEATMAP
# =========================
corr = numeric_df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10,8))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.show()