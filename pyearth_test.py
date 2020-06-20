import pandas as pd
from pyearth import Earth

df = pd.read_csv('pyearth_test.csv')
train = df[:-1]
model = Earth(max_degree=2)
model.fit(train[df.columns.difference(['value'])], train.value)
print(model.trace())
print(model.summary())
y_hat = model.predict(df[df.columns.difference(['value'])])
print(y_hat)

#import matplotlib.pyplot as plt
#plt.figure()
#plt.plot(train.value, 'b')
#plt.plot(y_hat, 'r')
#plt.show()
