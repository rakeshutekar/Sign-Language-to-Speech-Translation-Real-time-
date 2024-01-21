import json
import os

import matplotlib.pyplot as plt
import pandas as pd

# The directory where all the videos are stored
video_dir = "./WLASL/start_kit/raw_videos"

# Loading the JSON data from the WLASL_v0.3.json file
with open('./WLASL/start_kit/WLASL_v0.3.json') as f:
    WLASL_json_file_data = json.load(f)

# Creating an empty dataframe to store all the data
WLASL_df = pd.DataFrame()


for item in WLASL_json_file_data:
    # Extracting 'gloss' and 'instances' from each item
    gloss = item["gloss"]
    instances = item.get("instances", [])

    # Creating a dataframe from 'instances'
    instances_df = pd.json_normalize(instances) if instances else pd.DataFrame()

    if not instances_df.empty:
        # Adding 'gloss' column to the DataFrame
        instances_df['gloss'] = gloss

        # Looping over each row in order to check if the video file actually exists before adding it to WLASL_df
        for index, row in instances_df.iterrows():
            video_file = os.path.join(video_dir, row['video_id'] + '.mp4')
            if os.path.isfile(video_file):
                WLASL_df = WLASL_df._append(row, ignore_index=True)

# Sorting the entries alphabetically by 'gloss' before saving
WLASL_df = WLASL_df.sort_values(by='gloss')

# Putting 'gloss' as the first column and 'video_id' as the second
cols = ['gloss', 'video_id'] + [col for col in WLASL_df.columns if col not in ['gloss', 'video_id']]
WLASL_df = WLASL_df[cols]

# Converting to a CSV file
WLASL_df.to_csv('./WLASL/code/EDA/EDA_Outputs/EDA_Script_1_output.csv', index=False)

print(f"\nDataFrame head:\n{WLASL_df.head()}")
print(f"\nDataFrame columns:\n{WLASL_df.columns}")
print(f"\nDataFrame shape:\nRows, columns in the dataset: {WLASL_df.shape}")
print(f"\nTotal distinct words: {WLASL_df['gloss'].nunique()}")
print(f"\nTotal distinct signers: {WLASL_df['signer_id'].nunique()}")

# Initial Data Analysis Plots

# Plot 1
# Calculating the total number of videos for each gloss
gloss_counts = WLASL_df.groupby('gloss').size()
# Finding the top 10 glosses with the highest number of videos for the plot
top_10_glosses = gloss_counts.nlargest(10)

plt.figure(figsize=(10, 8))
ax = top_10_glosses.plot(kind='barh', color='orange', edgecolor='black')
for i, v in enumerate(top_10_glosses):
    ax.text(v - 1, i, str(v), color='black', ha='right')
ax.invert_yaxis()
plt.xlabel('Total Number of Videos')
plt.ylabel('Gloss (English word/phrase)')
plt.title('Top 10 Glosses with the Highest Number of Videos')
plt.savefig('./WLASL/code/EDA/EDA_Outputs/Top_10_Glosses.png', dpi=300, bbox_inches='tight')
plt.show()


# Plot 2
# Calculating the total number of videos signed by each signer
signer_counts = WLASL_df.groupby('signer_id').size()
# Finding the top 10 signers
top_10_signers = signer_counts.nlargest(10)

plt.figure(figsize=(10, 8))
ax = top_10_signers.plot(kind='bar', color='limegreen', edgecolor='black')
for i, v in enumerate(top_10_signers):
    ax.text(i, v + 4, str(v), color='black', ha='center')
plt.xlabel('Signer ID')
plt.xticks(rotation=0)
plt.ylabel('Number of Videos Signed')
plt.title('Top 10 Signers and Video Counts')
plt.savefig('./WLASL/code/EDA/EDA_Outputs/Top_10_Signers.png', dpi=300, bbox_inches='tight')
plt.show()
