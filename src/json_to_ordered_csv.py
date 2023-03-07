import json
from datetime import datetime 
import pandas as pd
from os import listdir

# Initialize an empty list to store dataframes
dfs = []

FILE_NAME = "data_leon"
FILE_LOCATION = 'data/{}/'.format(FILE_NAME)
files = [f for f in listdir(FILE_LOCATION) if f.startswith('endsong_')]
print(files)

# Opening all JSON file and return as dictionary
for file in files:
    print(FILE_LOCATION + file)
    with open(FILE_LOCATION + file, encoding='utf-8') as f:
        data = json.load(f)
        df_temp = pd.DataFrame.from_dict(data)
        dfs.append(df_temp)
#f.close()

# Merge dataframes to one dataframe
df = pd.concat(dfs, ignore_index=True)

print("Initial number of rows:", df.shape[0])
# Drop columns related to podcasts and browser
df.drop(['episode_name', 'episode_show_name', 'spotify_episode_uri', 'user_agent_decrypted', 'incognito_mode'], axis=1, inplace=True)

# Drop rows where there is no track uri
df.dropna(how='any', axis=0, subset=['spotify_track_uri'], inplace=True)
print("Afterwards number of rows:", df.shape[0])

# Sort the data by timestamp, example = 'ts': '2021-06-30T10:02:47Z'
df = df.sort_values(by=['ts'], ascending=True)

print(df.head())

# Export the merged DataFrame to a CSV file
df.to_csv('{}/{}.csv'.format(FILE_LOCATION, FILE_NAME), index=False)
