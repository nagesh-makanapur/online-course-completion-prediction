#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[2]:


# Load Dataset
df = pd.read_csv('online_course_completion.csv')
df.head()


# In[3]:


#Explore the Data
df.info()
df.describe()
df.isnull().sum()
df['completed_course'].value_counts()


# In[4]:


#Visualize the Data
sns.countplot(x='completed_course', data=df)
plt.title('Course Completion Distribution')
plt.show()
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()


# In[5]:


#Preprocess the Data
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    if col != 'completed_course':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
if df['completed_course'].dtype == 'object':
    df['completed_course'] = df['completed_course'].map({'yes':1, 'no':0})  # adjust based on values


# In[6]:


#Split the Data
X = df.drop('completed_course', axis=1)
y = df['completed_course']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


#Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[8]:


#Future Engineering 
if 'total_time_spent' in df.columns and 'num_sessions' in df.columns:
    df['avg_time_per_session'] = df['total_time_spent'] / (df['num_sessions'] + 1e-5)

if {'assignments_completed', 'assignments_given'}.issubset(df.columns):
    df['assignment_completion_rate'] = df['assignments_completed'] / (df['assignments_given'] + 1e-5)
    
for col in ['time_on_videos', 'total_time_spent']:
    if col in df.columns:
        df[col] = np.log1p(df[col])


# In[9]:


#Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)


# In[10]:


#Evaluate the Model
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))






