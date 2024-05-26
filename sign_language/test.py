import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split


df=pd.read_csv("sign_language\sign_mnist_train.csv")
train_df = df 
label = train_df['label']
print(train_df.shape)



plt.figure(figsize=(10, 6))
sns.countplot(x = label )
plt.title('Frequency of Each Label in Training Set')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.show()



train_df, test_df = train_test_split(train_df, test_size = 0.2,stratify=train_df['label'], random_state=42)
label_train = train_df['label']
label_test = test_df['label']
train_df = train_df.drop(['label'],axis = 1)
test_df = test_df.drop(['label'],axis = 1)

fig, axe = plt.subplots(1,2)
plt.suptitle('distribution of train and test')
sns.countplot(x=label_train, ax=axe[0])
axe[0].set_title('Train Set')
sns.countplot(x=label_test, ax=axe[1])
axe[1].set_title('Test Set')
plt.show() 

print(train_df.shape)
print(test_df.shape)