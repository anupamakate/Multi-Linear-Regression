#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Consider only the below columns and prepare a prediction model for predicting Price.

#Corolla<-Corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]

 

#Model -- model of the car
# Price  -- Offer Price in EUROs	
# Age_08_04 -- Age in months as in August 2004	
# Mfg_Month -- Manufacturing month (1-12)	
# Mfg_Year	-- Manufacturing Year
# KM -- Accumulated Kilometers on odometer
# Fuel_Type	 -- Fuel Type (Petrol, Diesel, CNG)
# HP -- Horse Power
# Met_Color	 -- Metallic Color?  (Yes=1, No=0)
# Color -- Color (Blue, Red, Grey, Silver, Black, etc.)
# Automatic	-- Automatic ( (Yes=1, No=0)
# cc -- Cylinder Volume in cubic centimeters
# Doors -- Number of doors
# Cylinders	-- Number of cylinders
# Gears -- Number of gear positions
# Quarterly_Tax -- Quarterly road tax in EUROs
# Weight -- Weight in Kilograms
# Mfr_Guarantee -- Within Manufacturer's Guarantee period  (Yes=1, No=0)
# BOVAG_Guarantee -- BOVAG (Dutch dealer network) Guarantee  (Yes=1, No=0)
# Guarantee_Period -- 	Guarantee period in months
# ABS -- Anti-Lock Brake System (Yes=1, No=0)
# Airbag_1 -- Driver_Airbag  (Yes=1, No=0)
# Airbag_2 -- Passenger Airbag  (Yes=1, No=0)
# Airco -- Airconditioning  (Yes=1, No=0)
# Automatic_airco -- Automatic Airconditioning  (Yes=1, No=0)
# Boardcomputer -- Boardcomputer  (Yes=1, No=0)
# CD_Player -- CD Player  (Yes=1, No=0)
# Central_Lock -- Central Lock  (Yes=1, No=0)
# Powered_Windows -- Powered Windows  (Yes=1, No=0)
# Power_Steering -- Power Steering  (Yes=1, No=0)
# Radio -- Radio  (Yes=1, No=0)
# Mistlamps	-- Mistlamps  (Yes=1, No=0)
# Sport_Model -- Sport Model  (Yes=1, No=0)
# Backseat_Divider -- Backseat Divider  (Yes=1, No=0)
# Metallic_Rim --Metallic Rim  (Yes=1, No=0)
# Radio_cassette -- Radio Cassette  (Yes=1, No=0)
# Tow_Bar -- Tow Bar  (Yes=1, No=0)


# In[10]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot


# In[11]:


# import dataset
toyo=pd.read_csv(r'C:\Users\anupa\Downloads\ToyotaCorolla.csv',encoding='latin1')
toyo


# In[12]:


#EDA
toyo.info()


# In[13]:


toyo2=pd.concat([toyo.iloc[:,2:4],toyo.iloc[:,6:7],toyo.iloc[:,8:9],toyo.iloc[:,12:14],toyo.iloc[:,15:18]],axis=1)
toyo2


# In[14]:


toyo3=toyo2.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)
toyo3


# In[15]:


toyo3[toyo3.duplicated()]


# In[16]:


toyo4=toyo3.drop_duplicates().reset_index(drop=True)
toyo4


# In[17]:


toyo4.describe()


# In[18]:


#Correlation Analysis
toyo4.corr()


# In[19]:


sns.set_style(style='darkgrid')
sns.pairplot(toyo4)


# In[20]:


#Model Building
model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyo4).fit()


# In[21]:


#Model Testing
# Finding Coefficient parameters
model.params


# In[22]:


# Finding tvalues and pvalues
model.tvalues , np.round(model.pvalues,5)


# In[23]:


# Finding rsquared values
model.rsquared , model.rsquared_adj   # Model accuracy is 86.17%


# In[24]:


# Build SLR and MLR models for insignificant variables 'CC' and 'Doors'
# Also find their tvalues and pvalues


# In[25]:


slr_c=smf.ols('Price~CC',data=toyo4).fit()
slr_c.tvalues , slr_c.pvalues # CC has significant pvalue


# In[26]:


slr_d=smf.ols('Price~Doors',data=toyo4).fit()
slr_d.tvalues , slr_d.pvalues # Doors has significant pvalue


# In[27]:


mlr_cd=smf.ols('Price~CC+Doors',data=toyo4).fit()
mlr_cd.tvalues , mlr_cd.pvalues # CC & Doors have significant pvalue


# In[28]:


#Model Validation Techniques
#Two Techniques: 1. Collinearity Check & 2. Residual Analysis
# 1) Collinearity Problem Check
# Calculate VIF = 1/(1-Rsquare) for all independent variables

rsq_age=smf.ols('Age~KM+HP+CC+Doors+Gears+QT+Weight',data=toyo4).fit().rsquared
vif_age=1/(1-rsq_age)

rsq_KM=smf.ols('KM~Age+HP+CC+Doors+Gears+QT+Weight',data=toyo4).fit().rsquared
vif_KM=1/(1-rsq_KM)

rsq_HP=smf.ols('HP~Age+KM+CC+Doors+Gears+QT+Weight',data=toyo4).fit().rsquared
vif_HP=1/(1-rsq_HP)

rsq_CC=smf.ols('CC~Age+KM+HP+Doors+Gears+QT+Weight',data=toyo4).fit().rsquared
vif_CC=1/(1-rsq_CC)

rsq_DR=smf.ols('Doors~Age+KM+HP+CC+Gears+QT+Weight',data=toyo4).fit().rsquared
vif_DR=1/(1-rsq_DR)

rsq_GR=smf.ols('Gears~Age+KM+HP+CC+Doors+QT+Weight',data=toyo4).fit().rsquared
vif_GR=1/(1-rsq_GR)

rsq_QT=smf.ols('QT~Age+KM+HP+CC+Doors+Gears+Weight',data=toyo4).fit().rsquared
vif_QT=1/(1-rsq_QT)

rsq_WT=smf.ols('Weight~Age+KM+HP+CC+Doors+Gears+QT',data=toyo4).fit().rsquared
vif_WT=1/(1-rsq_WT)

# Putting the values in Dataframe format
d1={'Variables':['Age','KM','HP','CC','Doors','Gears','QT','Weight'],
    'Vif':[vif_age,vif_KM,vif_HP,vif_CC,vif_DR,vif_GR,vif_QT,vif_WT]}
Vif_df=pd.DataFrame(d1)
Vif_df


# In[29]:


# None variable has VIF>20, No Collinearity, so consider all varaibles in Regression equation


# In[30]:


# 2) Residual Analysis
# Test for Normality of Residuals (Q-Q Plot) using residual model (model.resid)
sm.qqplot(model.resid,line='q') # 'q' - A line is fit through the quartiles # line = '45'- to draw the 45-degree diagonal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[31]:


list(np.where(model.resid>6000))  # outliar detection from above QQ plot of residuals


# In[32]:


list(np.where(model.resid<-6000))


# In[33]:


# Test for Homoscedasticity or Heteroscedasticity (plotting model's standardized fitted values vs standardized residual values)

def standard_values(vals) : return (vals-vals.mean())/vals.std()  # User defined z = (x - mu)/sigma


# In[34]:


plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show()


# In[35]:


# Test for errors or Residuals Vs Regressors or independent 'x' variables or predictors 
# using Residual Regression Plots code graphics.plot_regress_exog(model,'x',fig)    # exog = x-variable & endog = y-variable


# In[36]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Age',fig=fig)
plt.show()


# In[37]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'KM',fig=fig)
plt.show()


# In[38]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'HP',fig=fig)
plt.show()


# In[39]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'CC',fig=fig)
plt.show()


# In[40]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Doors',fig=fig)
plt.show()


# In[41]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Gears',fig=fig)
plt.show()


# In[42]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'QT',fig=fig)
plt.show()


# In[43]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Weight',fig=fig)
plt.show()


# In[44]:


#Model Deletion Diagnostics (checking Outliers or Influencers)
#Two Techniques : 1. Cook's Distance & 2. Leverage value
# 1. Cook's Distance: If Cook's distance > 1, then it's an outlier
# Get influencers using cook's distance
(c,_)=model.get_influence().cooks_distance
c


# In[45]:


# Plot the influencers using the stem plot
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(toyo4)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[46]:


# Index and value of influencer where C>0.5
np.argmax(c) , np.max(c)


# In[47]:


# 2. Leverage Value using High Influence Points : Points beyond Leverage_cutoff value are influencers
fig,ax=plt.subplots(figsize=(20,20))
fig=influence_plot(model,ax = ax)


# In[48]:


# Leverage Cuttoff Value = 3*(k+1)/n ; k = no.of features/columns & n = no. of datapoints
k=toyo4.shape[1]
n=toyo4.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff


# In[49]:


toyo4[toyo4.index.isin([80])] 


# In[50]:


#Improving the Model
# Creating a copy of data so that original dataset is not affected
toyo_new=toyo4.copy()
toyo_new


# In[51]:


# Discard the data points which are influencers and reassign the row number (reset_index(drop=True))
toyo5=toyo_new.drop(toyo_new.index[[80]],axis=0).reset_index(drop=True)
toyo5


# In[52]:


#Model Deletion Diagnostics and Final Model
while model.rsquared < 0.90:
    for c in [np.max(c)>0.5]:
        model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyo5).fit()
        (c,_)=model.get_influence().cooks_distance
        c
        np.argmax(c) , np.max(c)
        toyo5=toyo5.drop(toyo5.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
        toyo5
    else:
        final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyo5).fit()
        final_model.rsquared , final_model.aic
        print("Thus model accuracy is improved to",final_model.rsquared)


# In[53]:


final_model.rsquared # Model Accuracy is increased to 90.02%


# In[57]:


toyo5


# In[58]:


#Model Predictions
# say New data for prediction is
new_data=pd.DataFrame({'Age':12,"KM":40000,"HP":80,"CC":1300,"Doors":4,"Gears":5,"QT":69,"Weight":1012},index=[0])
new_data


# In[59]:


# Manual Prediction of Price
final_model.predict(new_data)


# In[60]:


# Automatic Prediction of Price with 90.02% accurcy
pred_y=final_model.predict(toyo5)
pred_y


# In[ ]:




