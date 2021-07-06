#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import plot, iplot
import plotly.graph_objs as go


# In[3]:


df=pd.read_csv(r'/content/1603428153-5e748a2d5fc288e9f69c5f86.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.shape


# In[ ]:


len(df)


# In[ ]:


df.corr()


# In[ ]:


df.describe()


# In[ ]:


df.dtypes


# In[ ]:


df.isnull().sum()


# In[ ]:


missing_val_count_by_column = (df.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.info()


# In[4]:


df.plot(kind='scatter', x='market_value', y='fpl_value')
plt.show()


# In[ ]:


sns.clustermap(df.corr(),annot=True)


# In[ ]:


df.corr()['market_value']


# In[ ]:


df.corr()['market_value'].sort_values()


# In[5]:


df['club'].describe()


# In[6]:


clubs = tuple(set(df['club']))
print(clubs)


# In[7]:


value = []
for club in clubs:
    value.append(sum(df['market_value'].loc[df['club']==club]))


# In[12]:


keys= clubs
values=value

iplot({
    "data": [go.Bar(x=keys, y=values)],
    "layout": go.Layout(title="Market Value of players of each club")
})


# In[13]:


plt.subplots(figsize=(15,6))
sns.set_color_codes()
sns.distplot(df['age'], color = "R")
plt.xticks(rotation=90)
plt.title('Distribution of Premier League Players Age')
plt.show()


# In[15]:


#number of premier league player position
plt.subplots(figsize=(15,6))
sns.countplot('position',data=df,palette='bright',edgecolor=sns.color_palette('dark',7),order=df['position'].value_counts().index)
plt.xticks(rotation=90)
plt.title('number of premier league player position')
plt.show()


# In[16]:


#Top 10 players market value
dfmarketv = df.nlargest(10, 'market_value').sort_values('market_value',ascending=False)
plt.subplots(figsize=(15,6))
sns.barplot(x="name", y="market_value",  data=dfmarketv ,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('top 10 bigest market value of premier league player season 2017/2018')
plt.show()


# In[18]:


# Average market value
df_meanmv=pd.DataFrame(df.groupby(['club'])['market_value'].mean()).reset_index().sort_values('market_value',ascending=False)
plt.subplots(figsize=(15,6))
sns.barplot(x="club", y="market_value",data=df_meanmv,palette='cool',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Average of Market Value in Every Clubs')
plt.show()


# In[21]:


#most view player
dfview = df.nlargest(10, 'page_views').sort_values('page_views',ascending=False)
plt.subplots(figsize=(15,6))
sns.barplot(x="name", y="page_views",  data=dfview ,palette='bright',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('top 10 most viewed premier league player season 2017/2018')
plt.show()


# In[22]:


df2 = df[['age','page_views','fpl_points','fpl_value','market_value']].copy()
df2.head()


# In[24]:


sns.set(style="ticks", color_codes=True)
g = sns.pairplot(df2)
plt.show()


# In[26]:


#club with their average of fpl value
df_meanfv=pd.DataFrame(df.groupby(['club'])['fpl_value'].mean()).reset_index().sort_values('fpl_value',ascending=False)
plt.subplots(figsize=(15,6))
sns.barplot(x="club", y="fpl_value",data=df_meanfv,palette='cool',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Average of FPL Value in Every Clubs')
plt.show()


# In[32]:


missing_val_count_by_column = (df.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# In[33]:


df.dropna(inplace=True)


# In[34]:


df.info()


# In[36]:


df['big_club'].value_counts()
sns.countplot(x='big_club',data=df,palette='Set1')


# In[38]:


sns.jointplot(x='market_value',y='page_views',kind='reg',data=df,
              marginal_kws={
                            'color':'red'})


# In[39]:


df[(df['page_views']>7000) & (df['market_value']<30)]


# In[41]:


sns.lmplot(x='market_value', y='page_views', hue='big_club',palette="RdBu",markers='*', data=df[(df['page_views']<3000) & (df['market_value']<30)])


# In[42]:


df['fpl_sel']=df['fpl_sel'].apply(lambda x:float(x.split('%')[0]))
df['fpl_sel']


# In[43]:


df.corr()['market_value'].sort_values()


# In[44]:


plt.figure(figsize=(10,7))
sns.boxplot(x='position_cat',y='market_value',data=df)


# In[45]:


x=df[['position_cat','age','fpl_sel','fpl_points','page_views','fpl_value']]
y=df['market_value']


# In[46]:


sns.distplot(y)


# In[49]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)


# In[50]:


from sklearn.linear_model import LinearRegression


# In[51]:


lm = LinearRegression()
lm.fit(x_train,y_train)


# In[52]:


coeffecients = pd.DataFrame(lm.coef_,x.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[53]:


predictions = lm.predict( x_test)


# In[54]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[55]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[56]:


sns.distplot((y_test-predictions),bins=20);


# RIDGE REGRESSION

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt


# 

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)


# In[ ]:


print(x_train.shape); print(x_test.shape)


# In[ ]:


rr = Ridge(alpha=0.5)
rr.fit(x_train, y_train) 
pred_train_rr= rr.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print(r2_score(y_train, pred_train_rr))

pred_test_rr= rr.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_rr))) 
print(r2_score(y_test, pred_test_rr))


# In[ ]:


coeffecients = pd.DataFrame(rr.coef_,x.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[ ]:


predictions = rr.predict( x_test)


# In[57]:


#function is used to fill NA/NaN values using the specified method
df['region'] = df['region'].fillna(2.0) 


# In[ ]:





# ### *DATA PROCESSING TO INCREASE CLEAN DATASET*

# In[64]:


data = pd.read_csv('/content/1603428153-5e748a2d5fc288e9f69c5f86.csv')
data.head()


# In[65]:


#function is used to fill NA/NaN values using the specified method
data['region'] = data['region'].fillna(2.0)


# In[66]:


# Data preprocessing 

class PreProcessing():

  def __init__(self, data):
    self.drop_columns = ['name', 'club_id', 'age_cat', ]  #We only need numeric columns for our models 
                                                          #so we are deleting unnecessary data
    self.dummy_columns = ['club', 'position', 'position_cat', 'region'
                          , 'nationality']
    self.data = data

  #Assigning position to specific code
  def correct_pos_cat(self,single_pos_cat):
    if single_pos_cat == 1:
      return "attackers"
    elif single_pos_cat == 2:
      return "midfielders"
    elif single_pos_cat == 3:
      return "defenders"
    else:
      return "goalkeepers"

  #Assigning region to specific code
  def correct_region(self, single_region):
    if single_region == 1:
      return "England"
    elif single_region == 2:
      return "EU"
    elif single_region == 3:
      return "Americas"
    else:
      return "Rest of World"

  #Converted Float entries into Num entries
  def correct_fpl_sel(self):
    for i in self.data['fpl_sel'].index:                            
      self.data['fpl_sel'][i] = float(self.data['fpl_sel'][i][:-1])
    self.data['fpl_sel'] = pd.to_numeric(self.data['fpl_sel'])

  def log_trans(self):    
    '''
       Finding the Log value
       This is done to reduce the explosive dependencies otherwise creating 
       alot if dependent data
    '''
    self.data['page_views'] = self.data['page_views'].apply(np.log)

  def preprocessed_data(self):
    self.data = self.data.drop(self.drop_columns, axis=1) #Axis: 0 for columns; and 1 for rows
    self.data['region'] = self.data['region'].apply(self.correct_region)
    self.data['position_cat'] = self.data['position_cat'].apply(self.correct_pos_cat)
    self.correct_fpl_sel()
    self.log_trans()
    self.data = pd.get_dummies(self.data, self.dummy_columns, drop_first=True)

    return self.data

data


# In[67]:


preprocessor = PreProcessing(data)
encoded_data = preprocessor.preprocessed_data()

encoded_data.head()


# In[68]:


#TRAIN AND SPLIT DATASET
output_var = 'market_value'
X_encode = encoded_data[encoded_data.columns[~encoded_data.columns.isin([output_var])]]
y_encode = encoded_data[output_var]


from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(X_encode, test_size=0.25, random_state=3)

y_train = y_encode.loc[x_train.index.values]
y_test = y_encode.loc[x_test.index.values]
x_train = X_encode.loc[x_train.index.values,:]
x_test = X_encode.loc[x_test.index.values,:]

x_train


# In[69]:


#SCALING
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

y_train


# # MODELS

# ### **LINEAR REGRESSION**

# In[70]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std


linear_reg = LinearRegression()
linear_reg.fit(x_train_scaled,y_train)

train_score = linear_reg.score(x_train_scaled, y_train)
test_score = linear_reg.score(x_test_scaled, y_test)

y_pred = linear_reg.predict(x_test_scaled)
train_predict = linear_reg.predict(x_train_scaled)

print("Linear Regression:")
print()
print("Train R2 score: ", train_score)
print("Test R2 score: ", test_score)
print()

# The mean squared error
print('Training - Mean squared error: %.2f' 
      % mean_squared_error(y_train, train_predict))

# The mean squared error
print('Test - Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))


# ### **RIDGE REGRESSOR**
# 
# 
# 
# 

# In[71]:


from sklearn.linear_model import Ridge
ridge_reg = Ridge()
ridge_reg.fit(x_train_scaled, y_train)

train_score = ridge_reg.score(x_train_scaled, y_train)
test_score = ridge_reg.score(x_test_scaled, y_test)
print("Accuracy on training data - " + str(train_score))
print("Accuracy on test data - " + str(test_score))
print()

train_pred = ridge_reg.predict(x_train_scaled) 
test_pred = ridge_reg.predict(x_test_scaled)
train_mse = mean_squared_error(y_train,train_pred)
test_mse = mean_squared_error(y_test,test_pred)
print("Mean Square Error on training data - " + str(train_mse))
print("Mean Square Error on test data - " + str(test_mse))


# ### **LASSO REGRESSION**

# In[77]:


from sklearn.linear_model import Lasso
lasso_reg = Lasso(normalize=True)
lasso_reg.fit(x_train_scaled, y_train)
train_score = lasso_reg.score(x_train_scaled, y_train)
test_score = lasso_reg.score(x_test_scaled, y_test)
print("Accuracy on training data - " + str(train_score))
print("Accuracy on test data - " + str(test_score))
print()
train_pred = lasso_reg.predict(x_train_scaled) 
test_pred = lasso_reg.predict(x_test_scaled)
train_mse = mean_squared_error(y_train,train_pred)
test_mse = mean_squared_error(y_test,test_pred)
print("Mean Square Error on training data - " + str(train_mse))
print("Mean Square Error on test data - " + str(test_mse))


# ### **RANDOM FOREST**

# In[72]:


from sklearn.ensemble import RandomForestRegressor

np.random.seed(13)
rand_forest = RandomForestRegressor()
rand_forest.fit(x_train_scaled, y_train)
train_score = rand_forest.score(x_train_scaled, y_train)
results = np.mean(cross_val_score(rand_forest, x_train_scaled, y_train, 
                                  scoring="r2"))

test_score = rand_forest.score(x_test_scaled, y_test)


y_pred = rand_forest.predict(x_test_scaled)

print("Random Forest Regresser:")
print("Train score: ", train_score)
print("Train score with cross validation: ", results)

print("Test score: ", test_score)

# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

#R square
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))


# In[ ]:





# ### **SUPPORT VECTOR REGRESSOR**

# In[73]:


from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


np.random.seed(2)
# svr = SVR(C=100, epsilon=0.01, kernel='rbf')
# svr = SVR(C =100.0, epsilon=0.1, kernel='poly', degree=2)
svr = SVR()



svr.fit(x_train_scaled, y_train)
train_score = svr.score(x_train_scaled, y_train)
test_score = svr.score(x_test_scaled, y_test)

train_pred = svr.predict(x_train_scaled)
test_pred = svr.predict(x_test_scaled)

print("Support vector regressor -\n")
print("Training scrore - ", train_score)
print("Test score - ", test_score)
print()
print("Training - Mean squared error : ", mean_squared_error(train_pred, y_train))
print("Testing - Mean squared error : ", mean_squared_error(test_pred, y_test))


# ### **KNN**

# In[81]:


from sklearn.neighbors import KNeighborsRegressor
neigh =  KNeighborsRegressor(n_neighbors=5)
neigh.fit(x_train_scaled,y_train)
train_score = neigh.score(x_train_scaled, y_train)
test_score = neigh.score(x_test_scaled, y_test)

train_pred = neigh.predict(x_train_scaled)
test_pred = neigh.predict(x_test_scaled)

print("Nearest Neighbor  -\n")
print("Training scrore - ", train_score)
print("Test score - ", test_score)
print()
print("Training - Mean squared error : ", mean_squared_error(train_pred, y_train))
print("Testing - Mean squared error : ", mean_squared_error(test_pred, y_test))


# ## **HYPERPARAMETER**

# In[89]:


from sklearn.svm import SVR

np.random.seed(2)
# svr = SVR(C=100, kernel='rbf', gamma=0.1, epsilon=1)
svr = SVR(C =100.0, epsilon=0.1, kernel='poly', degree=2)

svr.fit(x_train_scaled, y_train)
train_score = svr.score(x_train_scaled, y_train)
test_score = svr.score(x_test_scaled, y_test)

results = np.mean(cross_val_score(svr, x_train_scaled, y_train, 
                                  scoring="r2"))

train_pred = svr.predict(x_train_scaled)
test_pred = svr.predict(x_test_scaled)

print("Support vector regressor -\n")
print("Training scrore - ", train_score)
print("Test score - ", test_score)
print()
print("Training - Mean squared error : ", mean_squared_error(train_pred, y_train))
print("Testing - Mean squared error : ", mean_squared_error(test_pred, y_test))


# In[90]:


from scipy.stats import uniform, truncnorm, randint

svr_model_params = {
    # randomly sample numbers from 4 to 204 estimators
    'degree':[2,3,4,5],
    # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
    'kernel': ['linear', 'poly', 'rbf'],
    # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
    'gamma': uniform(0.001, 0.1), 
    "C" : uniform(1, 1000), 
    "epsilon":uniform(0.1, 1)
}


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

# create gradient Boosting regressor model
svr_model =SVR()

# set up random search meta-estimator
# this will train 100 models over 5 folds of cross validation (500 models total)
reg = RandomizedSearchCV(svr_model, svr_model_params, n_iter=100, cv=5, random_state=42)

# train the random search meta-estimator to find the best model out of 100 candidates
model_gb = reg.fit(x_train_scaled, y_train)


# print winning set of hyperparameters
from pprint import pprint
pprint(model_gb.best_estimator_.get_params())


# In[ ]:


model_gb.score(x_train_scaled, y_train)
model_gb.score(x_test_scaled, y_test)


# ### **GENETIC ALGORITHM**

# In[83]:


size = 460

# this is the size of breeding pool and used to generate the population
n=25

# in making this is a triangle number we can breed 
# the top n individuals and get the same size population back
pop_size= sum(range(n+1))
print(pop_size)


# In[84]:


def eval_fit(pop):
    fit_vals = []
    for i in range(len(pop)):
        fit_vals.append(np.sum(pop[i]))
        
    return np.array(fit_vals)


# In[85]:


def rank_pop(pop):
    ranked =  [ pop[i] for i in np.argsort(-eval_fit(pop))]
    return ranked


# In[86]:


#CROSSOVER
def cross_pop(pop):
    new_pop = []
    for i in range(n):
        for j in range(i,n):
            x = np.random.randint(low=int(size/4),high=3*int(size/4)) # crossover point between 1/4 and 3/4
            new_pop.append(np.concatenate([pop[i][:x],pop[j][x:]]))
    return new_pop


# In[87]:


# MUTATION

def mut_pop(pop,k):       # 1/k is prob of mutating an individual
    for i in range(len(pop)):
        x = np.random.randint(0,k)
        if(x==0):
            y = np.random.randint(0,size)
            pop[i][y] = (pop[i][y]+1) %2
    return pop


# In[88]:


# creates a population
pop = []

for i in range(pop_size):    
    pop.append(np.random.randint(low=0,high=2, size=(size)))

    
# runs the algorithm and finds an optimum
m = 0
mut_prob = 3   # probability of a mutation in a given individual (i.e. 1/mut_prob)
best_fitn = np.amax(eval_fit(pop))

while(best_fitn < size and m<100):
        
    pop = rank_pop(pop)
    pop = cross_pop(pop)
    pop = mut_pop(pop,mut_prob)
    
    print("Generation: " + str(m))
    print(str(best_fitn) + " : " + str(100*best_fitn/size) + "%")
    #print(pop[0])

    best_fitn = np.amax(eval_fit(pop))
    m=m+1
  
print("\n")
print("Completed at generation: " + str(m))
print("Best fitness is: " + str(100*best_fitn/size) + "%")
pop = rank_pop(pop)
print("Best individual is: ")
pop[0]


# In[ ]:




