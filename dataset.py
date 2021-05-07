import pandas as pd


regex = '(\d*),Walking,(\d*),(.*),(.*),(.*)(,);'


file = open('dataset.txt', 'r', encoding='utf-8')
lines = file.readlines()
count = 0

#Lettura file csv (comma-separated values):

headers = ['user', 'activity', 'timestamp', 'x-accel', 'y-accel', 'z-accel']
df = pd.read_csv('dataset.txt', names=headers)

df_t = df.drop(columns=['user', 'timestamp'])
for i in range(0, len(df_t)):
    print(df_t.loc[[i]])
#DataFrame solo con colonne selezionate (a parità di righe):
