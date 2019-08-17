#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance, to_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


# In[2]:


####import dataset
df=pd.read_csv("fraud.csv") 


# In[3]:


#####this one is same as sample 
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig',                         'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})


# In[32]:


df1=pd.read_csv("fraud.csv") 
df1=df1.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig',                         'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})


# In[34]:


df1['errorBalanceOrig']=round(df1['newBalanceOrig']+df1['amount']-df1['oldBalanceOrig'])
df1['errorBalanceDest']=round(df1['oldBalanceDest']+df1['amount']-df1['newBalanceDest'])


# In[37]:


df1['nameDest'].value_counts()


# In[28]:


C


# In[11]:


import collections


# In[26]:


B=collections.Counter(C)
B


# In[14]:


np.unique(B)


# In[20]:


C1=np.sum(B)


# In[24]:


count


# In[4]:


######all data informations 
def datainfo(df):
    print("the first 5 row of dataframe is :")
    print(df.head())
    print('\n\n')
    print('##################################################################################################')
    print("the type of entitines  of dataframe are :")
    df.info()
    print('\n\n')
    print('##################################################################################################')
    print("the statistical information of dataframe are :")
    print('\n\n')
    print(df.describe())
    print('\n\n')
    print('##################################################################################################')
    print("the index  of dataframe are :")
    print('\n\n')
    print(df.index)
    print('\n\n')
    print('##################################################################################################')
    print("the column's names of dataframe are :")
    print('\n\n')
    print(df.columns)
    print('\n\n')
    print('##################################################################################################')
    print("the shape of dataframe is:")
    print('\n\n')
    print(df.shape)
    print('\n\n')
    print('##################################################################################################')
    print("Test if there any missing values in DataFrame:")
    print('\n\n')
    print(df.isnull().values.any())
    


# In[275]:


datainfo(df)


# In[5]:


print('\nAre there any merchants among originator accounts for CASH_IN transactions? {}'.format((df.loc[df.type == 'CASH_IN'].nameOrig.str.contains('M')).any())) 


# In[7]:


print('\nAre there any merchants among destination accounts for CASH_OUT transactions? {}'.format((df.loc[df.type == 'CASH_OUT'].nameDest.str.contains('M')).any())) # False


# In[8]:


print('\nAre there merchants among any originator accounts? {}'.format(      df.nameOrig.str.contains('M').any())) # False

print('\nAre there any transactions having merchants among destination accounts other than the PAYMENT type? {}'.format((df.loc[df.nameDest.str.contains('M')].type != 'PAYMENT').any())) # False


# In[ ]:


###############Exploratory Data Analysis


# In[9]:


def fraudulent(dataframe,column1,column2,column3,action1,action2,value1,value2):
    global Fraud_Transfer
    global Fraud_Cashout
    global Transfer
    global Flagged 
    global Not_Flagged
    global Not_Fraud
    print('\n The types of fraudulent transactions are ')
    print(dataframe.loc[dataframe[column1]== value1].type.drop_duplicates().values)
    print('\nThe type of transactions in which isFlaggedFraud is set:\{}'.format(list(dataframe.loc[dataframe[column3] == value1].type.drop_duplicates())))
    Fraud_Transfer = dataframe.loc[(dataframe[column1] == value1) & (dataframe[column2] == action1)]
    Fraud_Cashout = dataframe.loc[(dataframe[column1] == value1) & (dataframe[column2] == action2)]
    Transfer = dataframe.loc[dataframe[column2]== action1]
    Flagged = dataframe.loc[dataframe[column3] == value1]
    Not_Flagged = dataframe.loc[dataframe[column3] == value2]
    Not_Fraud = dataframe.loc[dataframe.isFraud == value2]
    print ('\n The number of fraudulent TRANSFERs = {}'.format(len(Fraud_Transfer))) 
    print ('\n The number of fraudulent CASH_OUTs = {}'.format(len(Fraud_Cashout))) 
    print('\n the Minimum amount transacted when isFlaggedFraud is:{}'.format(Flagged.amount.min()))
    print('\n the Maximum  amount transacted in a TRANSFER where isFlaggedFraud is :{}'.format(Transfer.loc[Transfer[column3] == 0].amount.max()))
    


# In[10]:


fraudulent(df,'isFraud','type','isFlaggedFraud','TRANSFER','CASH_OUT',1,0)


# In[11]:


def Destination():
    global Fraudulent_Dest
    Dest_Transfer=Fraud_Transfer.nameDest.isin(Fraud_Cashout.nameOrig).any()
    Fraudulent_Dest=Fraud_Transfer.loc[Fraud_Transfer.nameDest.isin(Not_Fraud.loc[Not_Fraud.type == 'CASH_OUT'].nameOrig.drop_duplicates())]
    IsOrgin_Fraud=Flagged.nameOrig.isin(pd.concat([Not_Flagged.nameOrig,Not_Flagged.nameDest])).any()
    IsDest_Orgin=Flagged.nameDest.isin(Not_Flagged.nameOrig).any()
    DestCount=sum(Flagged.nameDest.isin(Not_Flagged.nameDest))
    print('\nHave originators of transactions flagged as fraud transacted more than once?',IsOrgin_Fraud)
    print('\nHave destinations for transactions flagged as fraud initiated other transactions?',IsDest_Orgin)
    print('\nWithin fraudulent transactions, are there destinations for TRANSFERS that are also originators for CASH_OUTs?',Dest_Transfer)
    print('\nFraudulent TRANSFERs whose destination accounts are originators of genuine CASH_OUTs: \n\n',Fraudulent_Dest)
    print('\nHow many destination accounts of transactions flagged as fraud have been destination accounts more than once?:',DestCount)


# In[12]:


Destination()


# In[13]:


def tranStep(stepid):
    global step_cashout
    global step_transfer
    step_transfer=df[df.nameDest==stepid].step.values
    step_cashout=Not_Fraud.loc[(Not_Fraud.type == 'CASH_OUT') & (Not_Fraud.nameOrig ==stepid)].step.values
    print('Fraudulent TRANSFER',stepid,' occured at step:',step_transfer,'whereas genuine CASH_OUT from this account occured earlier at step :',step_cashout)


# In[40]:


a='C423543548'
tranStep(a)
    


# In[13]:


def FlaggedBalance():
    global F_balanced
    F_balanced=len(Transfer.loc[(Transfer.isFlaggedFraud == 0) & (Transfer.oldBalanceDest == 0) & (Transfer.newBalanceDest == 0)])
    print('\nThe number of TRANSFERs where isFlaggedFraud = 0, yet oldBalanceDest = 0 and\newBalanceDest = 0:',F_balanced)


# In[14]:


FlaggedBalance()


# In[15]:


def minmaxbalance():
    global minold
    global maxold
    global minnew
    global maxnew
    minold=Flagged.oldBalanceOrig.min()
    maxold=Flagged.oldBalanceOrig.max()
    minnew=Transfer.loc[(Transfer.isFlaggedFraud == 0)&(Transfer.oldBalanceOrig== Transfer.newBalanceOrig)].oldBalanceOrig.min()
    maxnew=Transfer.loc[(Transfer.isFlaggedFraud == 0) & (Transfer.oldBalanceOrig == Transfer.newBalanceOrig)].oldBalanceOrig.max()
    print('\nMinimum and Maximum of oldBalanceOrig for isFlaggedFraud = 1 TRANSFERs:  {}'.format([round(minold),round(maxold)]))
    print('\nMinimum and Maximum of oldBalanceOrig for isFlaggedFraud = 0 TRANSFERs where oldBalanceOrig =newBalanceOrig: {}'.format([round(minnew),round(maxnew)]))


# In[16]:


minmaxbalance()


# In[60]:


#######################Data cleaning###############################


# In[166]:


def cleaning(data,column1,column2,column3,column4,column5,column6,column7,column8,column9):
    global X
    global Y
    global Xfraud
    global XnonFraud
    global fractF
    global fractG
    randomState = 5
    np.random.seed(randomState)
    X = data.loc[(data[column1] == 'TRANSFER') | (data[column1] == 'CASH_OUT')]
    Y = X[column5]
    del X[column5]
    X = X.drop([column2, column3, column4], axis = 1)
    X.loc[X[column1] == 'TRANSFER', 'type'] = 0
    X.loc[X[column1] == 'CASH_OUT', 'type'] = 1
    X[column1] = X[column1].astype(int)
    ################## Imputation of Latent Missing Values
    Xfraud = X.loc[Y == 1]
    XnonFraud = X.loc[Y == 0]
    fractF=len(Xfraud.loc[(Xfraud[column6] == 0) & (Xfraud[column7] == 0) & (Xfraud.amount)])/(1.0 * len(Xfraud))
    fractG=len(XnonFraud.loc[(XnonFraud[column6] == 0) & (XnonFraud[column7] == 0) & (XnonFraud.amount)])/(1.0 * len(XnonFraud))
    X.loc[(X[column6] == 0) & (X[column7] == 0) & (X.amount != 0),[column6,column7]] = - 1
    X.loc[(X[column8] == 0) & (X[column9] == 0) & (X.amount != 0),[column8,column9]] = np.nan
    print('####################################################################################')
    print('\nThe fraction of fraudulent transactions with \'oldBalanceDest\' = \'newBalanceDest\' = 0 although the transacted \'amount\' is non-zero is: ',format(fractF))
    print('####################################################################################') 
    print('\nThe fraction of genuine transactions with \'oldBalanceDest\' = \newBalanceDest\' = 0 although the transacted \'amount\' is non-zero is: ',format(fractG))
    print('####################################################################################')
    
  


# In[167]:


cleaning(df,'type','nameOrig', 'nameDest', 'isFlaggedFraud','isFraud','oldBalanceDest','newBalanceDest','oldBalanceOrig', 'newBalanceOrig')


# In[21]:


X.head()


# In[23]:


#######################Feature Engineering ###############################


# In[19]:


def Fengineering(column1,column2,column3,column4,column5):
    global newcol1
    global newcol2
    newcol1= X[column1] + X[column2]- X[column3]
    newcol2= X[column4] + X[column2 ] - X[column5]
Fengineering('newBalanceOrig','amount','oldBalanceOrig','oldBalanceDest','newBalanceDest')
X['errorBalanceOrig']=newcol1
X['errorBalanceDest']=newcol2
X.head()


# In[31]:


#3FINDING HOURS AND DAYS/copy the dataset to new dataset,then make the new dataset and cange type to new name
data_new = df.copy()
# initializing feature column
data_new["type1"]=np.nan 
# filling feature column
data_new.loc[df1.nameOrig.str.contains('C') & df1.nameDest.str.contains('C'),"type1"] = "CC" 
data_new.loc[df1.nameOrig.str.contains('C') & df1.nameDest.str.contains('M'),"type1"] = "CM"
data_new.loc[df1.nameOrig.str.contains('M') & df1.nameDest.str.contains('C'),"type1"] = "MC"
data_new.loc[df1.nameOrig.str.contains('M') & df1.nameDest.str.contains('M'),"type1"] = "MM"


# In[33]:


# Subsetting data into observations with fraud and valid transactions:
fraud = data_new[data_new["isFraud"] == 1]
valid = data_new[data_new["isFraud"] == 0]
fraud = fraud.drop('type1', 1)
valid = valid.drop('type1',1)
data_new = data_new.drop('type1',1)


# In[ ]:


####assume that transaction only occur when transaction type is either CASH_OUT or TRANSFER.
valid = valid[(valid["type"] == "CASH_OUT")| (valid["type"] == "TRANSFER")]
data_new = data_new[(data_new["type"] == "CASH_OUT") | (data_new["type"] == "TRANSFER")]


# In[35]:


fraud = data_new[data_new["isFraud"] == 1]
valid = data_new[data_new["isFraud"] == 0]


# In[38]:


###omitting the nameOrig and nameDest columns from analysis.
names = ["nameOrig","nameDest"]
fraud = fraud.drop(names, 1)
valid = valid.drop(names,1)
data_new = data_new.drop(names,1)


# In[39]:


#######omitting the isFlaggedFraud column from the analysis
fraud = df1[df1["isFraud"] == 1]
valid = df1[df1["isFraud"] == 0]
fraud = fraud.drop("isFlaggedFraud",1)
valid = valid.drop("isFlaggedFraud",1)


# In[63]:


###defining function to seee number of valid/fraud  transaction over time 
def fraudhist(X) :  
    bins = 60
    valid.hist(column="step",color="red",bins=bins)
    plt.xlabel("1 hour time step")
    plt.ylabel("# of transactions")
    plt.title("# of valid transactions over time")
    X.hist(column ="step",color="yellow",bins=bins)
    plt.xlabel("1 hour time step")
    plt.ylabel("# of transactions")
    plt.title("# of fraud transactions over time")
    plt.tight_layout()
    plt.show()


# In[64]:


fraudhist(fraud)


# In[160]:


# getting hours and days of the week
def DayHour(X,Y):
    global fraud_days
    global fraud_hours
    global valid_days
    global valid_hours
    num_days = 7
    num_hours = 24
    fraud_days = X.step % num_days
    fraud_hours = X.step % num_hours
    valid_days = Y.step % num_days
    valid_hours = Y.step % num_hours
    # plotting scatterplot of the days of the week, identifying the fraudulent transactions (red) from the valid transactions (yellow) 
    plt.subplot(2, 2, 1)
    fraud_days.hist(bins=num_days,color="red")
    plt.title('Fraud transactions by Day')
    plt.xlabel('Day of the Week')
    plt.ylabel("# of transactions")
    plt.subplot(2, 2, 2)
    valid_days.hist(bins=num_days,color="yellow")
    plt.title('Valid transactions by Day')
    plt.xlabel('Day of the Week')
    plt.ylabel("# of transactions")
    plt.subplot(2, 2, 3)
    fraud_hours.hist(bins=num_hours, color="blue")
    plt.title('Fraud transactions by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel("# of transactions")
    plt.subplot(2, 2, 4)
    valid_hours.hist(bins=num_hours, color="magenta")
    plt.title('Valid transactions by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel("# of transactions")
    plt.tight_layout()
    plt.savefig('foo.png')
    plt.show()


# In[161]:


DayHour(fraud,valid)


# In[51]:


dataset1 = data_new.copy()


# adding feature HourOfDay to Dataset1 
dataset1["Dayofweek"] = np.nan 
dataset1["HourOfDay"] = np.nan # initializing feature column
dataset1.HourOfDay = data_new.step % 24
dataset1["Dayofweek"] = np.nan # initializing feature column
dataset1.Dayofweek = data_new.step % 7

print("Head of dataset1: \n", pd.DataFrame.head(dataset1))


# In[62]:


alpha = 0.3
fig,ax = plt.subplots()
valid.plot.scatter(x="step",y="amount",color="red",alpha=alpha,ax=ax,label="Valid Transactions")
fraud.plot.scatter(x="step",y="amount",color="yellow",alpha=alpha,ax=ax, label="Fraudulent Transactions")

plt.title("1 hour timestep vs amount")
plt.xlabel("1 hour time-step")
plt.ylabel("amount moved in transaction")
plt.legend(loc="upper right")

# plotting a horizontal line to show where valid transactions behave very differently from fraud transactions

plt.axhline(y=10000000)
plt.show()


print("Proportion of transactions where the amount moved is greater than 10 million: ",       len(data_new[data_new.amount > 10000000])/len(data_new))


# In[45]:


#######################Data visualization###############################


# In[58]:


limit = len(X)

def plotStrip(x, y, hue, figsize = (14, 9)):
    
    fig = plt.figure(figsize = figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    with sns.axes_style('ticks'):
        ax = sns.stripplot(x, y,hue = hue, jitter = 0.4, marker = '.',size = 4, palette = colours)
        ax.set_xlabel('')
        ax.set_xticklabels(['genuine', 'fraudulent'], size = 16)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, ['Transfer', 'Cash out'], bbox_to_anchor=(1, 1),                loc=2, borderaxespad=0, fontsize = 16);
    return ax


# In[62]:


ax = plotStrip(Y[:limit], X.step[:limit], X.type[:limit])
ax.set_ylabel('time [hour]', size = 6)
ax.set_title('Striped vs. homogenous fingerprints of genuine and fraudulent transactions over time', size = 10);

fig,ax=plt.subplot(2, 2, 1)
ax = plotStrip(Y[:limit], X.step[:limit], X.type[:limit])
ax.set_ylabel('time [hour]', size = 6)
ax.set_title('Striped vs. homogenous fingerprints of genuine and fraudulent transactions over time', size = 10)
fig,ax=plt.subplot(2, 2, 2)
ax = plotStrip(Y[:limit], X.amount[:limit], X.type[:limit], figsize = (14, 9))
ax.set_ylabel('amount', size = 16)
ax.set_title('Same-signed fingerprints of genuine and fraudulent transactions over amount', size = 18);


# In[52]:


limit = len(X)
ax = plotStrip(Y[:limit], X.amount[:limit], X.type[:limit], figsize = (14, 9))
ax.set_ylabel('amount', size = 16)
ax.set_title('Same-signed fingerprints of genuine and fraudulent transactions over amount', size = 18);


# In[64]:


limit = len(X)
ax = plotStrip(Y[:limit], - X.errorBalanceDest[:limit], X.type[:limit],               figsize = (14, 9))
ax.set_ylabel('- errorBalanceDest', size = 16)
ax.set_title('Opposite polarity fingerprints over the error in destination account balances', size = 18);


# In[71]:


def plot3d(df1,df2,x,y,z,zOffset,limit):
    sns.reset_orig() # prevent seaborn from over-riding mplot3d defaults
    fig = plt.figure(figsize = (10, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df1.loc[df2 == 0, x][:limit], df1.loc[df2 == 0, y][:limit], -np.log10(df1.loc[df2== 0, z][:limit] + zOffset), c = 'b', marker = '.', s = 1, label = 'genuine')
    ax.scatter(df1.loc[df2== 1, x][:limit], df1.loc[df2 == 1, y][:limit],  -np.log10(df1.loc[df2 == 1, z][:limit] + zOffset), c = 'y', marker = '.', s = 1, label = 'fraudulent')
    ax.set_xlabel(x, size = 16); 
    ax.set_ylabel(y + ' [hour]', size = 16); 
    ax.set_zlabel('- log$_{10}$ (' + z + ')', size = 16)
    ax.set_title('Error-based features separate out genuine and fraudulent transactions', size = 20)
    plt.axis('tight')
    ax.grid(1)
    noFraudMarker = mlines.Line2D([], [], linewidth = 0, color='b', marker='.',markersize = 10, label='genuine')
    fraudMarker = mlines.Line2D([], [], linewidth = 0, color='y', marker='.',markersize = 10, label='fraudulent')
    plt.legend(handles = [noFraudMarker, fraudMarker],bbox_to_anchor = (1.20, 0.38 ), frameon = False, prop={'size': 16});


# In[72]:


plot3d(X,Y,'errorBalanceDest','step','errorBalanceOrig',0.02,len(X))


# In[20]:


def updateFraud(df1,df2):
    global x_fraud
    global x_nonfraud
    global corre_non_fraud
    global indices 
    global mask
    global corre_fraud
    x_fraud = df1.loc[df2== 1] 
    x_nonfraud = df1.loc[df2 == 0]
    corre_non_fraud = x_nonfraud.loc[:, df1.columns != 'step'].corr()
    corre_fraud = x_fraud.loc[:, df1.columns != 'step'].corr()
    mask = np.zeros_like( corre_non_fraud)
    indices = np.triu_indices_from( corre_non_fraud)
    mask[indices] = True
    Skew_fraud=len(x_fraud) / float(len(df1))
    print('####################################################################################')
    print('print head of x_fraud with cleaned data: ',x_nonfraud.head())
    print('####################################################################################')
    print('show the correlation heatmap for x_fraud with cleaned data\n ')
    print('print head of x_fraud with cleaned data: ',x_fraud.head())
    print('####################################################################################\n')
    print('show the Detect Fraud in Skewed Data: ',Skew_fraud)
   
    
   
    
    
    


# In[21]:


updateFraud(X,Y)


# In[213]:


print('show the correlation heatmap for Non-fraud with cleaned data\n ')
corre_non_fraud.style.background_gradient(cmap='coolwarm')


# In[95]:


corre_Non_Fraud.style.background_gradient(cmap='coolwarm')


# In[163]:


def plotfraud(df1):
       grid_kws = {"width_ratios": (.9, .9, .05), "wspace": 0.2}
       f,(ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws,figsize = (14, 9))
       cmap =sns.cubehelix_palette(8)
       ax1 =sns.heatmap(corre_non_fraud, ax = ax1, vmin = -1, vmax = 1,cmap = cmap, square = False, linewidths = 0.5, mask = mask, cbar = False)
       ax1.set_xticklabels(ax1.get_xticklabels(), size = 16); 
       ax1.set_yticklabels(ax1.get_yticklabels(), size = 16); 
       ax1.set_title('Genuine \n transactions', size = 20)
       ax2 = sns.heatmap(corre_fraud, vmin = -1, vmax = 1, cmap = cmap, ax = ax2, square = False, linewidths = 0.5, mask = mask, yticklabels = False,cbar_ax = cbar_ax, cbar_kws={'orientation': 'vertical', 'ticks': [-1, -0.5, 0, 0.5, 1]})
       ax2.set_xticklabels(ax2.get_xticklabels(), size = 16); 
       ax2.set_title('Fraudulent \n transactions', size = 20);
       cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size = 14);


# In[148]:


plotfraud(X)


# In[176]:


dataset1.head()


# In[235]:


def updateFraud1(df1,df2):
    global x_fraud1
    global x_nonfraud1
    global corre_non_fraud1
    global indices1 
    global mask1
    global corre_fraud1
   
    x_fraud1 = df1.loc[df2== 1] 
    x_nonfraud1 = df1.loc[df2 == 0]
    corre_non_fraud1 = x_nonfraud1.loc[:, df1.columns != 'step'].corr()
    
    corre_fraud1 = x_fraud1.loc[:, df1.columns != 'step'].corr()
    mask1 = np.zeros_like( corre_non_fraud1)
    indices1 = np.triu_indices_from( corre_non_fraud1)
    mask1[indices1] = True
    Skew_fraud1=len(x_fraud1) / float(len(df1))
    print('####################################################################################')
    print('print head of x_fraud with cleaned data: ',x_nonfraud1.head())
    print('####################################################################################')
    print('show the correlation heatmap for x_fraud with cleaned data\n ')
    print('print head of x_fraud with cleaned data: ',x_fraud1.head())
    print('####################################################################################\n')
    print('show the Detect Fraud in Skewed Data: ',Skew_fraud1)
   


# In[236]:


X1 = dataset1.loc[(dataset1['type'] == 'TRANSFER') | (dataset1['type'] == 'CASH_OUT')]
Y1 = X1['isFraud']
del X1['isFraud']
X1 = X1.drop(['isFlaggedFraud'], axis = 1)
X1.loc[X1['type'] == 'TRANSFER', 'type'] = 0
X1.loc[X1['type'] == 'CASH_OUT', 'type'] = 1
X1['type'] = X1['type'].astype(int)


# In[237]:


updateFraud1(X1,Y1)


# In[238]:


def plotfraud1(df1):
       grid_kws = {"width_ratios": (.9, .9, .05), "wspace": 0.2}
       f,(ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws,figsize = (14, 9))
       cmap =sns.cubehelix_palette(8)
       ax1 =sns.heatmap(corre_non_fraud1, ax = ax1, vmin = -1, vmax = 1,cmap = cmap, square = False, linewidths = 0.5, mask = mask1, cbar = False)
       ax1.set_xticklabels(ax1.get_xticklabels(), size = 16); 
       ax1.set_yticklabels(ax1.get_yticklabels(), size = 16); 
       ax1.set_title('Genuine \n transactions', size = 20)
       ax2 = sns.heatmap(corre_fraud1, vmin = -1, vmax = 1, cmap = cmap, ax = ax2, square = False, linewidths = 0.5, mask = mask1, yticklabels = False,cbar_ax = cbar_ax, cbar_kws={'orientation': 'vertical', 'ticks': [-1, -0.5, 0, 0.5, 1]})
       ax2.set_xticklabels(ax2.get_xticklabels(), size = 16); 
       ax2.set_title('Fraudulent \n transactions', size = 20);
       cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size = 14);


# In[239]:


plotfraud1(X1)


# In[22]:


X.fillna(X.mean(), inplace=True)
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2,                                                 random_state = 42)


# In[ ]:


clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic')
param_dist = {'n_estimators': stats.randint(150, 1000),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4]
             }

numFolds = 5
kfold_5 = cross_validation.KFold(n = len(X), shuffle = True, n_folds = numFolds)

clf = RandomizedSearchCV(clf_xgb, 
                         param_distributions = param_dist,
                         cv = kfold_5,  
                         n_iter = 5, # you want 5 here not 25 if I understand you correctly 
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 3, 
                         n_jobs = -1)


# In[29]:


def xgbClassifier (df1, df2,a,b,n):
    global Xgb
    trainX, testX, trainY, testY = train_test_split(df1, df2, test_size = a,random_state = 42)
    weights = (df2 == 0).sum() / (1.0 * (df2 == 1).sum())
    Xgb= XGBClassifier(max_depth =b, scale_pos_weight = weights,n_jobs =n)
    probabilities = Xgb.fit(trainX, trainY).predict_proba(testX)
    print('AUPRC = {}'.format(average_precision_score(testY,probabilities[:, 1])))
   
    
    


# In[30]:


xgbClassifier(X,Y,0.2,3,4)


# In[105]:


fig = plt.figure(figsize = (14, 9))
ax = fig.add_subplot(111)

colours = ["r", "g", "b", "peachpuff", "orange","gray","yellow","m","c"]

ax = plot_importance(Xgb, height = 1, color = colours, grid = False,                      show_values = False, importance_type = 'cover', ax = ax);
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
ax.set_xlabel('importance score', size = 16);
ax.set_ylabel('features', size = 16);
ax.set_yticklabels(ax.get_yticklabels(), size = 12);
ax.set_title('Ordering of features by importance to the model learnt', size = 20);


# In[106]:


to_graphviz(Xgb)


# In[ ]:


#################Bias-variance tradeoff


# In[23]:


def bias(K,Z,K_train,Z_train,a,b):
   global trainScoresMean
   global trainScoresStd 
   global crossValScoresMean
   global crossValScoresStd
   global trainSizes
   global trainScoresStd 
   global crossValScores
   global trainScores 
   weights = (Z == 0).sum() / (1.0 * (Z == 1).sum())
   trainSizes, trainScores, crossValScores = learning_curve(XGBClassifier(max_depth =a, scale_pos_weight = weights, n_jobs = b), K_train,Z_train, scoring = 'average_precision')
   trainScoresMean = np.mean(trainScores, axis=1)
   trainScoresStd = np.std(trainScores, axis=1)
   crossValScoresMean = np.mean(crossValScores, axis=1)
   crossValScoresStd = np.std(crossValScores, axis=1)
   print('#####################trainScoresMean##################################')
   print(trainScoresMean)
   print('#####################trainScoresStd##################################')
   print(trainScoresStd)
   print('#####################crossValScoresMean##################################')
   print(crossValScoresMean)
   print('#####################crossValScoresStd##################################')
   print(crossValScoresStd)


# In[24]:


bias(X,Y,trainX,trainY,3,4)


# In[89]:


def biasGragh() :
   colours = plt.cm.tab10(np.linspace(0, 1, 9))
   fig = plt.figure(figsize = (14, 9))
   plt.fill_between(trainSizes, trainScoresMean - trainScoresStd,
   trainScoresMean + trainScoresStd, alpha=0.1, color=colours[0])
   plt.fill_between(trainSizes, crossValScoresMean - crossValScoresStd,
   crossValScoresMean + crossValScoresStd, alpha=0.1, color=colours[1])
   plt.plot(trainSizes, trainScores.mean(axis = 1), 'o-', label = 'train', color = colours[0])
   plt.plot(trainSizes, crossValScores.mean(axis = 1), 'o-', label = 'cross-val',color = colours[1])
   ax = plt.gca()
   for axis in ['top','bottom','left','right']:
     ax.spines[axis].set_linewidth(2)
   handles, labels = ax.get_legend_handles_labels()
   plt.legend(handles, ['train', 'cross-val'], bbox_to_anchor=(0.8, 0.15),loc=2, borderaxespad=0, fontsize = 16);
   plt.xlabel('training set size', size = 16); 
   plt.ylabel('AUPRC', size = 16)
   plt.title('Learning curves indicate slightly underfit model', size = 20);


# In[90]:


biasGragh()


# In[ ]:


###########################logistic


# In[25]:


# ####Hyperparameters
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
X1_train,X1_test,y1_train,y1_test=train_test_split(X,Y,test_size=0.25,random_state=0)
logreg = LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=5)
logreg_cv.fit(X,Y)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# In[ ]:


def EvaluationMetrics(K,Z,z_test,z_train,z_pred):
   print('####################################################################################')
   # Model Accuracy, how often is the classifier correct?
   print("Accuracy:",metrics.accuracy_score(z_test, z_pred))
   print('####################################################################################')
   print("Precision:",metrics.precision_score(z_test, z_pred))
   print('####################################################################################')
   print("Recall:",metrics.recall_score(z_test, z_pred))
   print('#####################confusion_matrix########################################')
   print(confusion_matrix(z_test, z_pred))
   print('#####################classification_report########################################')
   print(classification_report(z_test, z_pred))
   print('#####################Cross validation kfold=5 ########################################')
   print(cross_val_score(logreg, X,Y,cv=5) )


# In[72]:


def ROC(K,Z,z_test, z_pred_prob):   
    fpr, tpr, thresholds = roc_curve(z_test, z_pred_prob)
    # create plot
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    _ = plt.xlabel('False Positive Rate')
    _ = plt.ylabel('True Positive Rate')
    _ = plt.title('ROC Curve')
    _ = plt.xlim([-0.02, 1])
    _ = plt.ylim([0, 1.02])
    _ = plt.legend(loc="lower right")


# In[62]:


# fit the model with data
logreg.fit(X1_train,y1_train)
y_pred1=logreg.predict(X1_test)
y_pred_prob1 = gnb.predict_proba(X1_test)[:,1]
y_pred_prob1 = gnb.predict_proba(X1_test)[:,1]


# In[69]:


EvaluationMetrics(X,Y,y1_test,y1_train,y_pred1)  


# In[93]:


ROC(X,Y,y1_test, y_pred_prob1)


# In[ ]:


###################Navie


# In[112]:


####Hyperparameters
tuned_parameters = { 'tfidf__use_idf': (True, False),'tfidf__norm': ('l1', 'l2'),'alpha': [1, 1e-1, 1e-2]}
gnb = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3,random_state=109)
gnb_cv=GridSearchCV(gnb,tuned_parameters,cv=5)
gnb_cv.fit(X,Y)

print("tuned hpyerparameters :(best parameters) ",gnb_cv.best_params_)
print("accuracy :",gnb_cv.best_score_)


# In[80]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3,random_state=109) # 70% training and 30% test
#Train the model using the training sets
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
y_pred_prob = gnb.predict_proba(X_test)[:,1]


# In[94]:


EvaluationMetrics(X,Y,y_test,y_train,y_pred)  


# In[108]:


ROC(X,Y,y_test, y_pred_prob)

