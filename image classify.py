import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score

Categories = ['cats', 'dogs']
flat_data_arr = []
target_arr = []

datadir = '.../animal/'

for i in Categories:
    print(f'Loading category:{i}')
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_arr = imread(os.path.join(path, img))
        img_resized = resize(img_arr, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded Category:{i} successfully')
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

df = pd.DataFrame(flat_data)
df['Target'] = target
print(df.shape)

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.0001, 0.001, 0.01, 1],
    'kernel': ['rbf', 'poly']
}

svc = svm.SVC()

model = GridSearchCV(svc, param_grid)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print(f'The model accuracy is {accuracy * 100}%')