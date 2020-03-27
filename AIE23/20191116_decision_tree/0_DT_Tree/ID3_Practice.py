import numpy as np
import pandas as pd
import math
print(math.log(0,2))
tennis = pd.read_csv("data/tennis_decision.csv")
print(tennis.columns)
print(tennis.groupby(
    "Decision").Decision.count())  # equivalent to: select dicision,count(*) from tennis group by decision
#####################################################
# Step 1. Calculate Entropy for decision H(D)
#####################################################
D = -1 * 5 / 14 * np.log2(5 / 14) - 1 * 9 / 14 * np.log2(9 / 14)
print("H(D):", D)
print('\n------------------------------------')
#####################################################
# Step 2. Wind factor on decision g(D,A) = H(D)-H(D|A)
#####################################################
windcount = tennis.groupby(["Wind", "Decision"])["Wind", "Decision"].count()
print(windcount.rename_axis(["Wind_L", "Decision_L"]).sort_values(by=["Wind_L"], ascending=False))
weekwind = -1 * 2 / 8 * np.log2(2 / 8) - 1 * 6 / 8 * np.log2(6 / 8)  # Entropy(Decision|Wind=Week)
strongwind = -1 * 3 / 6 * np.log2(3 / 6) - 1 * 3 / 6 * np.log2(3 / 6)  # Entropy(Decision|Wind=Strong)
gwind = D - (8 / 14 * weekwind + 6 / 14 * strongwind)
print("gwind:", gwind)
print('\n------------------------------------')

#####################################################
# Step 3. Other factors on decision
#####################################################
# humicount = tennis.groupby(["Humidity", "Decision"])["Humidity"].count()
# print(humicount.rename_axis(["Humidity_L", "Decision_L"]))
highhumi = -1 * 4 / 7 * np.log2(4 / 7) - 1 * 3 / 7 * np.log2(3 / 7)  # Entropy(Decision|Humidity=High)
normalhumi = -1 * 1 / 7 * np.log2(1 / 7) - 1 * 6 / 7 * np.log2(6 / 7)  # Entropy(Decision|Humidity=Normal)
ghumi = D - (7 / 14 * highhumi + 7 / 14 * normalhumi)
print("ghumi:", ghumi)

# tempcount = tennis.groupby(["Temp.", "Decision"])["Temp."].count()
# print(tempcount.rename_axis(["Temp_L", "Decision_L"]))
cooltemp = -1 * 1 / 4 * np.log2(1 / 4) - 1 * 3 / 4 * np.log2(3 / 4)  # Entropy(Decision|Temp=Cool)
hottemp = -1 * 2 / 4 * np.log2(2 / 4) - 1 * 2 / 4 * np.log2(2 / 4)  # Entropy(Decision|Temp=Hot)
mildtemp = -1 * 2 / 6 * np.log2(2 / 6) - 1 * 4 / 6 * np.log2(4 / 6)  # Entropy(Decision|Temp=Mild)
gtemp = D - (4 / 14 * cooltemp + 4 / 14 * hottemp + 6 / 14 * mildtemp)
print("gtemp:", gtemp)

# Outlookcount = tennis.groupby(["Outlook", "Decision"])["Outlook"].count()
# print(Outlookcount.rename_axis(["Outlook_L", "Decision_L"]))
OvercastOutlook = -1 * 4 / 4 * np.log2(4 / 4)  # Entropy(Decision|Outlook=Overcast)
RainOutlook = -1 * 2 / 5 * np.log2(2 / 5) - 1 * 3 / 5 * np.log2(3 / 5)  # Entropy(Decision|Outlook=Rain)
SunnyOutlook = -1 * 3 / 5 * np.log2(3 / 5) - 1 * 2 / 5 * np.log2(2 / 5)  # Entropy(Decision|Outlook=Sunny)
gOutlook = D - (4 / 14 * OvercastOutlook + 5 / 14 * RainOutlook + 5 / 14 * SunnyOutlook)
print("gOutlook:", gOutlook)
print('\n------------------------------------')
#####################################################
# outlook decision appears in the root node of the tree
#                         Outlook
#                  /         |        \
#            Sunny(?)  Overcast(Yes)  Rain(?)
#####################################################

#####################################################
# Step 4 Calculate gain on rest features for Sunny day
#####################################################
# Gain(Outlook=Sunny|Wind)
sunnyOutlook = -1 * 3 / 5 * np.log2(3 / 5) - 1 * 2 / 5 * np.log2(2 / 5)  # Entropy(Outlook=Sunny)
SunnyStrongWind = -1 * 1 / 2 * np.log2(1 / 2) - 1 * 1 / 2 * np.log2(1 / 2)  # Entropy(Outlook=Sunny|Wind=Strong)
SunnyWeakWind = -1 * 2 / 3 * np.log2(2 / 3) - 1 * 1 / 3 * np.log2(1 / 3)  # Entropy(Outlook=Sunny|Wind=Week)
gSunnyWind = sunnyOutlook - (2 / 5 * SunnyStrongWind + 3 / 5 * SunnyWeakWind)
print("gSunnyWind:", gSunnyWind)
# Gain(Outlook=Sunny|Humidity)
sunnyHumidity = -1 * 3 / 5 * np.log2(3 / 5) - 1 * 2 / 5 * np.log2(2 / 5)  # Entropy(Outlook=Sunny|Humidity)
sunnyHighHumidity = 0
sunnyNormalHumidity = 0
gSunnyHumidity = sunnyHumidity
print("gSunnyHumidity:", gSunnyHumidity)
# Gain(Outlook=Sunny|Temperature)
# tempcount = tennis[tennis["Outlook"] == "Sunny"].groupby(["Temp.", "Decision"])["Temp."].count()
# print(tempcount.rename_axis(["Temp_L", "Decision_L"]))
SunnyCoolTem = 0  # Entropy(Outlook=Sunny|Temp=Cool)
SunnyHotTem = 0  # Entropy(Outlook=Sunny|Temp=Hot)
SunnyMildTem = -1 * 1 / 2 * np.log2(1 / 2) - 1 * 1 / 2 * np.log2(1 / 2)  # Entropy(Outlook=Sunny|Temp=Mild)
gSunnyTemp = sunnyOutlook - (1 / 5 * SunnyCoolTem + 2 / 5 * SunnyHotTem + 2 / 5 * SunnyMildTem)
print("gSunnyTemp:", gSunnyTemp)

#####################################################
# Humidity decision appears in the Sunny branch node of the tree
#                         Outlook
#            Sunny /         |        \
#            Humidity  Overcast(Yes)  Rain(?)
#####################################################
print('\n------------------------------------')
#####################################################
# Step 5 Calculate gain on rest features for Rain day
#####################################################
# Gain(Outlook=Rain|Wind)
rainoutlook = -1 * 3 / 5 * np.log2(3 / 5) - 1 * 2 / 5 * np.log2(2 / 5)  # Entropy(Outlook = Rain)
rainWeekWind = 0  # Entropy(Outlook=Sunny|Wind=Week)
rainStrongWind = 0  # Entropy(Outlook=Sunny|Wind=Strong)
grainWind = rainoutlook
print("grainWind:", grainWind)
# Gain(Outlook=Rain|Temp)
# tempcount = tennis[tennis["Outlook"] == "Rain"].groupby(["Temp.", "Decision"])["Temp."].count()
# print(tempcount.rename_axis(["Temp_L", "Decision_L"]))
rainCoolTemp = -1 * 1 / 2 * np.log2(1 / 2) - 1 * 1 / 2 * np.log2(1 / 2)
rainMildTemp = -1 * 1 / 3 * np.log2(1 / 3) - 1 * 2 / 3 * np.log2(2 / 3)
grainTemp = rainoutlook - (2 / 5 * rainCoolTemp + 3 / 5 * rainMildTemp)
print("grainTemp:", grainTemp)

#####################################################
# Wind decision appears in the Rain branch node of the tree
#                         Outlook
#            Sunny /         |        \Rain
#            Humidity  Overcast(Yes)  Wind
#####################################################