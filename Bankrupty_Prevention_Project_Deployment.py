#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('bankruptcy-prevention.csv',sep=';')


# # EDA

# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.isna().sum()


# In[7]:


df.dtypes


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df['class'].value_counts()


# In[11]:


df['class'].unique()


# In[12]:


df['industrial_risk'].unique()


# In[13]:


df['industrial_risk'].dtypes


# # Column Rename

# In[14]:


df=df.rename({'industrial_risk':'Industrial Risk','management_risk':'Management Risk','financial_flexibility':'Financial Flexibility','credibility':'Credibility','competitiveness':'Competitiveness','operating_risk':'Operating Risk'},axis=1)


# In[15]:


df.head()


# In[16]:


# Convert "bankruptcy" to 1 and "non-bankruptcy" to 0
df['class'] = df['class'].replace({'bankruptcy': 1, 'non-bankruptcy': 0})


# In[17]:


df.head()


# In[18]:


df.tail()


# # Data Visualization

# In[19]:


target_counts = df['class'].value_counts()
print(target_counts)

#Plotting the counts
plt.figure(figsize=(8,6))
sns.countplot(x='class', data=df, palette='Set2', order=target_counts.index)  # Assuming 'class' is the column name
plt.title('Distribution of Target Variable')
plt.xlabel('Target Variable')
plt.ylabel('Count')
plt.show()


# In[20]:


# Set the style for the plots
sns.set(style="whitegrid")

# Select numeric columns for plotting
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Plot the distributions for each feature
for column in numeric_columns:
    plt.figure(figsize=(12, 5))

    # Plot the overall distribution
    plt.subplot(1, 2, 1)
    sns.histplot(df[column], kde=True, color='skyblue')
    plt.title(f'Overall Distribution - {column}')

    # Plot the distribution for bankruptcy and non-bankruptcy cases
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=column, hue='class', kde=True, multiple='stack', palette='pastel')
    plt.title(f'Distribution by Class - {column}')

    plt.tight_layout()
    plt.show()


# In[21]:


# Set the style for the plots
plt.figure(figsize=(10, 5))
sns.set(style="ticks")

# Select numeric columns for the pairplot
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Plot the pairplot
sns.pairplot(df, hue='class', vars=numeric_columns, palette='husl')
plt.suptitle('Pairplot of Features by Class', y=1.02)
plt.show()


# In[22]:


# Calculate the correlation matrix
correlation_matrix = df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Create a heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


# # Checking Outliers

# In[23]:


# Set the style for the plots
sns.set(style="whitegrid")

fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=False, sharey=False)

sns.boxplot(x='Industrial Risk', data=df, palette='crest', ax=axes[0])
sns.boxplot(x='Management Risk', data=df, palette='crest', ax=axes[1])
sns.boxplot(x='Financial Flexibility', data=df, palette='crest', ax=axes[2])
sns.boxplot(x='Credibility', data=df, palette='crest', ax=axes[3])
sns.boxplot(x='Competitiveness', data=df, palette='crest', ax=axes[4])
sns.boxplot(x='Operating Risk', data=df, palette='crest', ax=axes[5])


plt.tight_layout(pad=2.0)
plt.show()


# # Balancing The Imbalanced Target Samples

# In[24]:


from imblearn.over_sampling import SMOTE
X = df.drop('class', axis=1)
y = df['class']
smote=SMOTE()
X_smote,y_smote=smote.fit_resample(X,y)

# Concatenate the resampled features and target variable
df = pd.concat([X_smote, y_smote], axis=1)

# Check the balanced class distribution in the new DataFrame
print("Class distribution in df after synthetic minority oversampling(SMOTE):")
print(df['class'].value_counts())


# In[25]:


target_counts = df['class'].value_counts()
print(target_counts)

#Plotting the counts
plt.figure(figsize=(8,6))
sns.countplot(x='class', data=df, palette='Set2', order=target_counts.index)  # Assuming 'class' is the column name
plt.title('Distribution of Target Variable')
plt.xlabel('Target Variable')
plt.ylabel('Count')
plt.show()


# In[26]:


grouped_stats = df.groupby('class').describe()

# Display descriptive statistics
print("Descriptive Statistics for Balanced Dataset:")
print(grouped_stats)


# In[27]:


X_smote.shape


# In[28]:


y_smote.shape


# # Model Building

# In[29]:


X=df.drop('class',axis=1)
y=df['class']


# In[30]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_smote,y_smote,test_size=0.3,random_state=42)


# In[31]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[37]:


classifier=LogisticRegression()
classifier.fit(X_train,y_train)


# In[38]:


y_pred=classifier.predict(X_test)


# In[39]:


# Generate classification report
report = classification_report(y_test, y_pred, output_dict=True)
df_classification_report = pd.DataFrame(report).transpose()

# Print the DataFrame
print("Classification Report:")
print(df_classification_report)


# In[40]:


confusion_matrix(y_test, y_pred)


# In[41]:


accuracy = accuracy_score(y_test, y_pred)*100
accuracy


# In[42]:


# Calculate training and testing accuracies
train_accuracy = classifier.score(X_train, y_train) * 100
train_accuracy


# In[43]:


test_accuracy = classifier.score(X_test, y_test) * 100
test_accuracy


# In[44]:


# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_nb)*100)
print("Classification Report:")
print(classification_report(y_test, y_pred_nb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))


# In[45]:


# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rf)*100)
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))


# In[46]:


# SVC
svc = SVC()
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_svc)*100)
print("Classification Report:")
print(classification_report(y_test, y_pred_svc))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svc))


# In[47]:


# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_knn)*100)
print("Classification Report:")
print(classification_report(y_test, y_pred_knn))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))


# In[48]:


# Decision Trees
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_dt)*100)
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))


# In[49]:


# Voting Classifier
voting_clf = VotingClassifier(estimators=[ 
    ('nb', nb), 
    ('knn', knn), 
    ('dt', dt)], 
    voting='hard')
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_voting)*100)
print("Classification Report:")
print(classification_report(y_test, y_pred_voting))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_voting))


# In[50]:


models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Ensemble": VotingClassifier(estimators=[
        ('KNN', KNeighborsClassifier()),
        ('Naive Bayes', GaussianNB()),
        ('Decision Tree', DecisionTreeClassifier())
    ])
}

model_accuracies = {
    "Model": [],
    "Training_Accuracy": [],
    "Testing_Accuracy": [],
}

# Iterate over each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Calculate training and testing accuracies
    train_accuracy = model.score(X_train, y_train) * 100
    test_accuracy = model.score(X_test, y_test) * 100

    # Add results to the dictionary
    model_accuracies["Model"].append(name)
    model_accuracies["Training_Accuracy"].append(train_accuracy)
    model_accuracies["Testing_Accuracy"].append(test_accuracy)

# Create a DataFrame from the dictionary
df_accuracies = pd.DataFrame(model_accuracies)

# Display the DataFrame
print(df_accuracies.to_string())


# In[51]:


model_accuracies = {}

colors = sns.color_palette("Set2")

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy

# Plotting the accuracies
plt.figure(figsize=(10, 6))

bars = plt.bar(model_accuracies.keys(), model_accuracies.values(), color=colors)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval*100:.2f}%", ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[52]:


input_data=(0.0,1.0,1.0,1.0,1.0,0.5)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=classifier.predict(input_data_reshaped)
print(prediction)
# Output the prediction
if prediction[0] == 1:
    print("The Company is bankruptcy")
else:
    print("The Company is non-bankruptcy")


# In[53]:


import pickle


# In[54]:


# saving The Trained Model.
filename='trained_model.sav'
pickle.dump(classifier,open(filename,'wb'))


# In[55]:


# Loading The Saved Model.
loaded_model=pickle.load(open('trained_model.sav','rb'))


# In[56]:


input_data=(0.0,1.0,1.0,1.0,1.0,0.5)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
# Output the prediction
if prediction[0] == 1:
    print("The Company is bankruptcy")
else:
    print("The Company is non-bankruptcy")


# In[ ]:




