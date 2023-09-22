#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib as plt
from mlxtend.frequent_patterns import apriori, association_rules


# #  Task 1: merge 12 datasets 

# In[2]:


# we will make a list with the urls of the files of data 
url=['https://raw.githubusercontent.com/KeithGalli/Pandas-Data-Science-Tasks/master/SalesAnalysis/Sales_Data/Sales_April_2019.csv',
     'https://raw.githubusercontent.com/KeithGalli/Pandas-Data-Science-Tasks/master/SalesAnalysis/Sales_Data/Sales_August_2019.csv',
     'https://raw.githubusercontent.com/KeithGalli/Pandas-Data-Science-Tasks/master/SalesAnalysis/Sales_Data/Sales_December_2019.csv',
     'https://raw.githubusercontent.com/KeithGalli/Pandas-Data-Science-Tasks/master/SalesAnalysis/Sales_Data/Sales_February_2019.csv',
     'https://raw.githubusercontent.com/KeithGalli/Pandas-Data-Science-Tasks/master/SalesAnalysis/Sales_Data/Sales_January_2019.csv',
     'https://raw.githubusercontent.com/KeithGalli/Pandas-Data-Science-Tasks/master/SalesAnalysis/Sales_Data/Sales_July_2019.csv',
     'https://raw.githubusercontent.com/KeithGalli/Pandas-Data-Science-Tasks/master/SalesAnalysis/Sales_Data/Sales_June_2019.csv',
     'https://raw.githubusercontent.com/KeithGalli/Pandas-Data-Science-Tasks/master/SalesAnalysis/Sales_Data/Sales_March_2019.csv',
     'https://raw.githubusercontent.com/KeithGalli/Pandas-Data-Science-Tasks/master/SalesAnalysis/Sales_Data/Sales_May_2019.csv',
     'https://raw.githubusercontent.com/KeithGalli/Pandas-Data-Science-Tasks/master/SalesAnalysis/Sales_Data/Sales_November_2019.csv',
     'https://raw.githubusercontent.com/KeithGalli/Pandas-Data-Science-Tasks/master/SalesAnalysis/Sales_Data/Sales_October_2019.csv',
     'https://raw.githubusercontent.com/KeithGalli/Pandas-Data-Science-Tasks/master/SalesAnalysis/Sales_Data/Sales_September_2019.csv']
df = pd.DataFrame() # make an emoty data frame to append on it each file to make all data in one file
#here we will var: hold --> hold each file in each iteration then append it to the empty data frame 
for i in url: 
    hold=pd.read_csv(i,index_col=0)
    df=pd.concat([df,hold])


# In[3]:


df


# # Task 2: what is the best month sales? and how much was earned in this month? 

# In[4]:


df


# In[5]:


# fist we need to know the information about data
df.info() # column'order date' is read as an object we need to convert it to DATE object but may we have null values
#so that we need to clean it to avoid error in converting 


# In[6]:


df.isnull().sum() # it appears to be whole row to be null


# In[7]:


df=df.dropna(how='any') # to make the change appear in the original dataframe


# In[8]:


df # no of rows has been decreased


# In[9]:


df.isnull().sum() # to make sure tha no nulls


# In[10]:


df["Order Date"] = pd.to_datetime(df["Order Date"]) # error here that there is a value called "Order Date" in the col
#and to convert we shouldn't have any word in it so that we need to nglect all values which contain any word 
#----------------
#-->note if we used \w+ it will return the same values insted of that, we need to neglect
#what has appeared in the error "Order Date"


# In[11]:


df[df["Order Date"].str.contains('Order.+')] # to search for an string has first 'Order' then any thing after it


# In[12]:


# to neglect the wrong values we use --> ~
df=df[~df["Order Date"].str.contains('Order.+')]
df


# In[13]:


df["Order Date"] = pd.to_datetime(df["Order Date"])


# In[14]:


df.info() # to make sure of the date object


# In[15]:


#now we need to to extract each month from the date and year also
#in each value in the column we will apply the function lambda as each iteration
#we assigning the value to x and apply the function to x
df["Order Date"].apply(lambda x: x.month)


# In[16]:


df["month"]=df["Order Date"].apply(lambda x: x.month)
df["year"]=df["Order Date"].apply(lambda x: x.year)


# In[17]:


df # here there isnot a sales column so we will create it as sales=Quantity Ordered*Price Each


# In[18]:


# first we need to convert the values to int
df=df.astype({'Quantity Ordered':'float','Price Each':'float'})


# In[19]:


df['sales']=df['Quantity Ordered']*df['Price Each']


# In[20]:


df


# In[21]:


sales=df.groupby('month')['sales'].sum()


# In[22]:


sales=sales.sort_values(ascending=False)
sales


# In[23]:


fig = px.histogram(df, x='month',y='sales',barmode='group',text_auto=True,
                   color_discrete_sequence= px.colors.sequential.Viridis,nbins=12,
                   title='what is the best month sales? and how much was earned in this month?')
fig.update_layout(xaxis = dict(tickmode = 'linear'),bargap=0.2)
fig.show()


# # task 3: which city has the best sales?

# In[24]:


df.head()


# In[25]:


# this to extract the city and its state from addres and store it in a new column
df['Purchase Address'].apply(lambda x: x.split(',')[-2]+' '+x.split(',')[-1].split(" ")[1])


# In[26]:


df['city']=df['Purchase Address'].apply(lambda x: x.split(',')[-2]+' '+x.split(',')[-1].split(" ")[1])
df


# In[27]:


fig = px.pie(df, values='sales', names='city',title='which city has the best sales?',hole=.3)
fig.show()


# # task 4: what time should we display advertisments to maximize  sales?

# In[28]:


#here i should deal with the hours in the column 'Order Date'
df['hour']=df['Order Date'].apply(lambda x: x.hour)


# In[29]:


df.head()


# In[30]:


fig = px.histogram(df, x='hour',y='sales',barmode='group',text_auto=True,
                   color_discrete_sequence= px.colors.sequential.thermal,nbins=24,
                   title='what time should we display advertisments to maximize sales?\n10 to 15 and 17 to 21')
fig.update_layout(xaxis = dict(tickmode = 'linear'),bargap=0.2)
fig.show()


# In[31]:


hour_sales=pd.DataFrame(df.groupby('hour')['sales'].sum())


# In[32]:


fig = px.line(hour_sales, x=hour_sales.index.values, y="sales", title='what time should we display advertisments to maximize sales?\n10 to 15 and 17 to 21',markers=True)
fig.update_layout(xaxis = dict(tickmode = 'linear'))

fig.show()


#  # task 5: what products are most often sold together 

# In[33]:


#here to save in a new df the duplicated id because it contains more than one product
#keep false-->it is to save the counts of duplicated id
df.reset_index(inplace=True)# convert index to a column
df_dup=df[df['Order ID'].duplicated(keep =False)]
df_dup


# In[34]:


df_dupg=df_dup.groupby(['Order ID', 'Product'])['Quantity Ordered'].sum().unstack().reset_index().fillna(0).set_index('Order ID')
df_dupg.astype(int)


# In[35]:


def hot_encode(x):
    if(x<= 0):
        return 0
    if(x>= 1):
        return 1
    
df_dup_encoded=df_dupg.applymap(hot_encode)
df_dupg=df_dup_encoded


# In[36]:


frq_items = apriori(df_dupg, min_support = 0.1, use_colnames = True)
frq_items


# In[37]:


# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 2)
# rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
rules


# In[38]:


df_dup.head()


# In[39]:


product=df_dup.groupby('Order ID')['Product'].apply(lambda x: ','.join(x))


# In[40]:


product # here we deal with whole one string so we need to separate it by ',' to get all the combination and its counts


# In[41]:


product=product.apply(lambda x:x.split(','))
product


# In[42]:


product.values


# In[50]:


from collections import Counter
from itertools import combinations 
count=Counter()
for i in product: 
    x=sorted(i)
    count.update(combinations(x,2))

count.most_common()


# In[44]:


mostsold_together=pd.DataFrame({"no. of transaction": count})
mostsold_together=mostsold_together.sort_values(by='no. of transaction',ascending=False)
topmostsold_together=mostsold_together.head(10)


# In[45]:


topmostsold_together


# In[46]:


fig = go.Figure(data=[go.Pie(labels=topmostsold_together.index.values,title='<b>what products are most often sold together<b>' ,values=topmostsold_together['no. of transaction'],pull=[0,0,0,0,0,0,0,0,0,.5])])
fig.show()


# In[51]:


fig = px.pie(topmostsold_together, values='no. of transaction', color_discrete_sequence=px.colors.sequential.RdBu,names=topmostsold_together.index.values,title='what products are most often sold together?')
fig.show()


# # task 6: what is the bestselling product?

# In[52]:


sales=df.groupby("Product")["sales"].sum()
sales


# In[53]:


sales_sorted=pd.DataFrame(sales).sort_values(by='sales',ascending=False)
sales_sorted


# In[54]:


fig = px.bar(sales_sorted, x=sales_sorted.index.values,y='sales',barmode='group'
                   ,text_auto=True,color_continuous_scale='thermal',color='sales',
                   
                   title='what is the bestselling product?\n<b><i>Macbook Pro Laptop<i><b>' )
fig.update_layout(xaxis = dict(tickmode = 'linear'),bargap=0.2)
fig.show()

