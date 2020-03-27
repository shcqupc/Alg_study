import numpy as np
import pandas as pd

ds = pd.read_csv("data/DS_RT.csv")
# print(ds)
#####################################################
# Standard deviation of golf players
#####################################################
golfplayers_std = ds["Golf Players"].std(ddof=0)
print("golfplayers_std:", golfplayers_std)
# print(np.std(ds["Golf Players"]))
# v1 = ds["Golf Players"] - ds["Golf Players"].sum()/ds["Golf Players"].count()
# print(np.sqrt(np.sum(v1**2/14)))

#####################################################
#  outlook
#####################################################
# print(ds.groupby(["Outlook"])["Outlook"].count())
# Sunny
sunny_std = ds[ds["Outlook"] == "Sunny"].iloc[:, -1].std(ddof=0)
print("sunny_std:", sunny_std)
# Overcast
overcast_std = ds[ds["Outlook"] == "Overcast"].iloc[:, -1].std(ddof=0)
print("overcast_std:", overcast_std)
# Rain
rain_std = ds[ds["Outlook"] == "Rain"].iloc[:, -1].std(ddof=0)
print("rain_std:", rain_std)
# Weighted standard deviation for outlook
weighted_outlook_std = 5 / 14 * sunny_std + 5 / 14 * rain_std + 4 / 14 * overcast_std
print("weighted_outlook_std:", weighted_outlook_std)
# Standard deviation reduction for outlook
red_outlook = golfplayers_std - weighted_outlook_std
print("red_outlook:", red_outlook)

print('\n------------------------------------')
#####################################################
#  Temp.
#####################################################
# print(ds.groupby(["Temp."])["Temp."].count())
# Cool
cool_std = ds[ds["Temp."] == "Cool"].iloc[:, -1].std(ddof=0)
print("cool_std:", cool_std)
# Hot
hot_std = ds[ds["Temp."] == "Hot"].iloc[:, -1].std(ddof=0)
print("hot_std:", hot_std)
# Mild
mild_std = ds[ds["Temp."] == "Mild"].iloc[:, -1].std(ddof=0)
print("mild_std:", mild_std)
# Weighted standard deviation for Temp.
weighted_temp_std = 4 / 14 * cool_std + 6 / 14 * mild_std + 4 / 14 * hot_std
print("weighted_temp_std:", weighted_temp_std)
# Standard deviation reduction for temp
red_temp = golfplayers_std - weighted_temp_std
print("red_temp:", red_temp)

print('\n------------------------------------')
#####################################################
#  Humidity
#####################################################
# print(ds.groupby(["Humidity"])["Humidity"].count())
# High
high_std = ds[ds["Humidity"] == "High"].iloc[:, -1].std(ddof=0)
print("high_std:", high_std)
# Normal
normal_std = ds[ds["Humidity"] == "Normal"].iloc[:, -1].std(ddof=0)
print("normal_std:", normal_std)
# Weighted standard deviation for Humidity
weighted_humidity_std = 7 / 14 * high_std + 7 / 14 * normal_std
print("weighted_humidity_std:", weighted_humidity_std)
# Standard deviation reduction for temp
red_humidity = golfplayers_std - weighted_humidity_std
print("red_humidity:", red_humidity)

print('\n------------------------------------')
#####################################################
#  Wind
#####################################################
# print(ds.groupby(["Wind"])["Wind"].count())
# High
strong_std = ds[ds["Wind"] == "Strong"].iloc[:, -1].std(ddof=0)
print("strong_std:", strong_std)
# Normal
weak_std = ds[ds["Wind"] == "Weak"].iloc[:, -1].std(ddof=0)
print("weak_std:", weak_std)
# Weighted standard deviation for Wind
weighted_wind_std = 6 / 14 * strong_std + 8 / 14 * weak_std
print("weighted_wind_std:", weighted_wind_std)
# Standard deviation reduction for temp
red_wind = golfplayers_std - weighted_wind_std
print("red_wind:", red_wind)
print(min(red_outlook, red_temp, red_humidity, red_wind))

#####################################################
#  put outlook decision at the top of decision tree
#  then use same way to get:
#  1. Standard deviation reduction for sunny outlook and temperature = 4.18
#  2. Standard deviation reduction for sunny outlook and humidity = 3.33
#  3. Standard deviation reduction for sunny outlook and wind = 0.85
#  then get temperature as brunch for sunny outlook
#####################################################

