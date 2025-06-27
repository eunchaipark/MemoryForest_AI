

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


csv_path = 'C:/For_Model/data/metrics.csv'


df = pd.read_csv(csv_path)

df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

df = df.sort_values('datetime')

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['accuracy'], label='Accuracy', marker='o')
plt.plot(df['datetime'], df['f1'], label='F1 Score', marker='o')
plt.plot(df['datetime'], df['precision'], label='Precision', marker='o')
plt.plot(df['datetime'], df['recall'], label='Recall', marker='o')

plt.title('Model Performance Over Time')
plt.xlabel('Datetime')
plt.ylabel('Score')
plt.ylim(0.85, 1.0)
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()


plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))


plt.show()
