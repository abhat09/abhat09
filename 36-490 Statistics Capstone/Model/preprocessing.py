import pandas as pd
import numpy as np

# Preprocessing
################################################################################################

# load in the data and create full name column
raw_data = pd.read_csv("data_2022_2023.csv")

data = raw_data
data['LastName'] = data['LastName'].str.lower().str.capitalize()
# combine first and last name
data['FullName'] = data['FirstName'] + ' ' + data['LastName']

# making names consistent for top contenders in the simulation
data.replace("Melanie De jesus dos santos", "Melanie Jesus santos", inplace=True)
data.replace('Nola Rhianne Matthews', 'Nola Matthews', inplace=True)
data.replace("Joscelyn Michelle Roberson", "Joscelyn Roberson", inplace=True)
data.replace("Curran Michael Phillips", 'Curran Phillips', inplace=True)
data.replace("Khoi Alexander Young", 'Khoi Young', inplace=True)
data.replace("Yul Kyung Tae Moldauer", "Yul Moldauer", inplace=True)

data['Country'].replace("ENG", "GBR", inplace=True) #both GBR and ENG should be GBR
data['Country'].replace("GE1", "GER", inplace=True) #both GE1 and GER should be GER

data['Apparatus'].replace("hb", "HB", inplace=True) #hb should be HB for consistency

# combining VT1 and VT2 into VT
data['Apparatus'].replace({"VT1": "VT", "VT2": "VT"}, inplace=True)

# removing all rows with no name or no score
data = data.dropna(subset=['FullName', 'Score'])

# unique columns to keep
data = data[['Gender', 'Competition', 'Round', 'Country', 'Apparatus', 'Score', 'FullName']]
data = data.drop_duplicates()

data.to_csv('preprocess.csv', index=False)