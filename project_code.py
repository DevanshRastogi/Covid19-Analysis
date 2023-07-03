import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df1= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data1.csv")
df2= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data2.csv")
df3= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data3.csv")
df4= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data4.csv")
df5= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data5.csv")
df6= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data6.csv")
df7= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data7.csv")
df8= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data8.csv")
df9= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data9.csv")
df10= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data10.csv")
df11= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data11.csv")
df12= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data12.csv")
df13= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data13.csv")
df14= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data14.csv")
df15= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data15.csv")
df16= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data16.csv")
df17= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data17.csv")
df18= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data18.csv")
df19= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data19.csv")
df20= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data20.csv")
df21= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data21.csv")
df22= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data22.csv")
df23= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data23.csv")
df24= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data24.csv")
df25= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data25.csv")
df26= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data26.csv")
df27= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data27.csv")
df28= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data28.csv")
df29= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data29.csv")
df30= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data30.csv")
df31= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data31.csv")
df32= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data32.csv")
df33= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data33.csv")
df34= pd.read_csv("https://data.covid19india.org/csv/latest/raw_data34.csv")


df1.rename(columns={"Num cases":"Num Cases"})
df2.rename(columns={"Num cases":"Num Cases"})

df= df1.append([df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20])

DATE= df['Date Announced'].str.split('/', expand=True)
DATE.columns= ['Day', 'Month','Year']
DATE

df=pd.concat([df, DATE], axis=1)
df

fig = plt.figure(figsize = (10, 5))

# creating the bar plot
M.plot.bar(color ='maroon',
        width = 0.4)

plt.xlabel("Months")
plt.ylabel("Deceased")
plt.title("Deaths due to Covid19 Virus")
plt.show()


#Creating a Correlation Matrix
df= pd.read_csv('example2.csv')


df.rename(
    columns=({ 'new_cases': 'New Cases', 'new_vaccinations': 'New Vaccinations'}),
    inplace=True,
)
correlationMatrix  = df.loc[:,:].corr()
print(correlationMatrix)
seaborn.heatmap(correlationMatrix, annot=True)
plt.savefig('corr1.png', dpi=300, bbox_inches='tight')


data[data['Current Status']=='Deceased'].groupby('Detected State')['Num Cases'].sum().sort_values(ascending=False)

#To check with age group was infected the most
M=data.groupby('Age Bracket')['Num Cases'].sum().sort_values(ascending=False).head(10)


#Support Vector Regression

Day= data[data['Current Status']=='Hospitalized'].groupby(['Month', 'Day'])['Num Cases'].sum()
x= np.arange(len(Day))
y= Day.values
y=y.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
Sx= sc_X.fit_transform(x)
Sy= sc_y.fit_transform(y)

from sklearn.svm import SVR
reg= SVR(kernel='rbf')
reg.fit(Sx,Sy.ravel())

plt.scatter(Sx,Sy)
plt.plot(Sx, reg.predict(Sx), color='k', linewidth=5)
plt.show()

#To check Regression Score
plt.scatter(Sx,Sy)
plt.plot(Sx, reg.predict(Sx), color='k', linewidth=5)
plt.show()

# Time Series Data
# Using graph_objects
import plotly.graph_objects as go

import pandas as pd

df= pd.read_csv('case_series.csv')

fig = go.Figure([go.Scatter(x=df['Month'], y=df['Daily Confirmed'])])
fig.update_layout(title_text="Time Series Data of Daily Confirmed Cases")

fig.show()


# To set the plot size
df2= pd.read_csv('case_series2.csv')
plt.figure(figsize=(300, 800))

# using .plot method to plot stock prices.
# we have passed colors as a list
df2.plot(color=['orange', 'green'])

# adding title
plt.title('Time series Analysis of Daily Recovered vs Daily Deceased')

# adding label to x-axis
plt.xlabel('Month')

# adding legend.
plt.legend()

#Death and Healed Ratio
discharged = df["cases"] - df["active cases"]

if "healed" in df.columns:
    df.drop("healed", axis=1, inplace=True)
if "healed" not in df.columns:
    df.insert(loc=len(df.columns),
              column="healed",
              value=(discharged -df["deaths"]))

healed_ratio = df["healed"] * 100 / discharged
healed_ratio
death_ratio = df["deaths"] * 100 / discharged
death_ratio

outcome = {"death_ratio": death_ratio, "healed_ratio": healed_ratio}
outcome_df = pd.DataFrame(outcome)
outcome_df
outcome_df.plot()

plt.title("Death and Healed Ratio during Feb21-Apr21")

