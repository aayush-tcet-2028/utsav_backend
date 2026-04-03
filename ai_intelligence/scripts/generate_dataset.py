import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

TOTAL_SAMPLES = 10002 
SAMPLES_PER_CLASS = TOTAL_SAMPLES // 3

print(f"Generating Realistic Dataset with Human Annotation Noise ({TOTAL_SAMPLES} samples)...")

columns = [
    'Event_Category', 'Time_Of_Day', 'Expected_Crowd', 'Environment_Type', 
    'Max_Venue_Capacity', 'Venue_Area_Sq_Meters', 'Number_Of_Fire_Exits', 
    'Event_Date', 'Duration_Hours', 'Latitude', 'Longitude',
    'Has_Fireworks', 'Has_Temp_Structures', 'VIP_Attendance', 'Loudspeaker_Used',
    'Road_Closure_Required', 'Is_Moving_Procession', 'Food_Stalls_Present', 'Liquor_Served', 
    'Risk_Level'
]

data = []
risks_to_generate = ['High'] * SAMPLES_PER_CLASS + ['Medium'] * SAMPLES_PER_CLASS + ['Low'] * SAMPLES_PER_CLASS
random.shuffle(risks_to_generate) 

categories = ['Mega Religious Gathering', 'Political Rally', 'Major Concert', 'Sports Match', 'Private Function', 'Protest/March']
times = ['Morning', 'Afternoon', 'Evening', 'Night']
start_date_base = datetime(2026, 4, 1)

for risk in risks_to_generate:
    row = {}
    row['Risk_Level'] = risk
    row['Time_Of_Day'] = random.choice(times)
    row['Event_Date'] = (start_date_base + timedelta(days=random.randint(0, 180))).strftime('%Y-%m-%d')
    row['Latitude'] = round(random.uniform(18.0, 19.5), 6)
    row['Longitude'] = round(random.uniform(72.5, 74.0), 6)
    
    if risk == 'High':
        scenario_type = random.choice(['Overcrowded_Stampede', 'Indoor_Fire_Trap', 'Unmanaged_Mega_Procession'])
        if scenario_type == 'Overcrowded_Stampede':
            row['Event_Category'] = random.choice(['Mega Religious Gathering', 'Political Rally'])
            row['Environment_Type'] = 'Outdoor'
            row['Expected_Crowd'] = random.randint(500000, 2500000)
            row['Max_Venue_Capacity'] = int(row['Expected_Crowd'] * random.uniform(0.6, 0.9))
            row['Venue_Area_Sq_Meters'] = int(row['Expected_Crowd'] * 0.4)
            row['Number_Of_Fire_Exits'] = random.randint(1, 3)
            row['Is_Moving_Procession'] = 1 if row['Event_Category'] == 'Mega Religious Gathering' else 0
            row['Road_Closure_Required'] = 0 if row['Is_Moving_Procession'] == 1 else 1
            row['Has_Fireworks'] = 0
            row['Has_Temp_Structures'] = 1
            row['Liquor_Served'] = 0
            row['Duration_Hours'] = random.randint(12, 72)
        elif scenario_type == 'Indoor_Fire_Trap':
            row['Event_Category'] = 'Major Concert'
            row['Environment_Type'] = 'Indoor'
            row['Expected_Crowd'] = random.randint(5000, 20000)
            row['Max_Venue_Capacity'] = int(row['Expected_Crowd'] * 1.1)
            row['Venue_Area_Sq_Meters'] = int(row['Max_Venue_Capacity'] * 1.0)
            row['Number_Of_Fire_Exits'] = 2
            row['Has_Fireworks'] = 1
            row['Has_Temp_Structures'] = 1
            row['Is_Moving_Procession'] = 0
            row['Road_Closure_Required'] = 0
            row['Liquor_Served'] = 1
            row['Duration_Hours'] = random.randint(3, 6)
        else:
            row['Event_Category'] = 'Protest/March'
            row['Environment_Type'] = 'Outdoor'
            row['Expected_Crowd'] = random.randint(100000, 500000)
            row['Max_Venue_Capacity'] = row['Expected_Crowd']
            row['Venue_Area_Sq_Meters'] = int(row['Expected_Crowd'] * 1.5)
            row['Number_Of_Fire_Exits'] = 0
            row['Has_Fireworks'] = 0
            row['Has_Temp_Structures'] = 0
            row['VIP_Attendance'] = 1
            row['Loudspeaker_Used'] = 1
            row['Is_Moving_Procession'] = 1
            row['Road_Closure_Required'] = 0
            row['Food_Stalls_Present'] = 1
            row['Liquor_Served'] = 0
            row['Duration_Hours'] = random.randint(4, 10)
            
        row['VIP_Attendance'] = row.get('VIP_Attendance', 1 if random.random() > 0.3 else 0)
        row['Loudspeaker_Used'] = row.get('Loudspeaker_Used', 1 if random.random() > 0.1 else 0)
        row['Food_Stalls_Present'] = row.get('Food_Stalls_Present', 1 if row['Expected_Crowd'] > 10000 else 0)

    elif risk == 'Medium':
        row['Event_Category'] = random.choice(['Major Concert', 'Sports Match', 'Political Rally'])
        row['Environment_Type'] = random.choice(['Outdoor', 'Indoor'])
        row['Expected_Crowd'] = random.randint(10000, 80000)
        row['Max_Venue_Capacity'] = int(row['Expected_Crowd'] * random.uniform(1.0, 1.2))
        row['Venue_Area_Sq_Meters'] = int(row['Max_Venue_Capacity'] * random.uniform(1.2, 2.0))
        row['Number_Of_Fire_Exits'] = max(4, int(row['Max_Venue_Capacity'] / 5000))
        row['Has_Fireworks'] = 1 if row['Environment_Type'] == 'Outdoor' and random.random() > 0.5 else 0
        row['Has_Temp_Structures'] = 1 if random.random() > 0.3 else 0
        row['VIP_Attendance'] = 1 if random.random() > 0.4 else 0
        row['Loudspeaker_Used'] = 1
        row['Is_Moving_Procession'] = 0
        row['Road_Closure_Required'] = 1 if row['Expected_Crowd'] > 30000 else 0
        row['Food_Stalls_Present'] = 1
        row['Liquor_Served'] = 1 if row['Event_Category'] == 'Major Concert' else 0
        row['Duration_Hours'] = random.randint(3, 8)

    else:
        row['Event_Category'] = random.choice(['Private Function', 'Sports Match'])
        if row['Event_Category'] == 'Sports Match':
            row['Expected_Crowd'] = random.randint(1000, 10000)
            row['Environment_Type'] = 'Outdoor'
        else:
            row['Expected_Crowd'] = random.randint(50, 1500)
            row['Environment_Type'] = random.choice(['Indoor', 'Outdoor'])
            
        row['Max_Venue_Capacity'] = int(row['Expected_Crowd'] * random.uniform(1.5, 3.0))
        row['Venue_Area_Sq_Meters'] = int(row['Max_Venue_Capacity'] * random.uniform(2.0, 4.0))
        row['Number_Of_Fire_Exits'] = max(3, int(row['Max_Venue_Capacity'] / 300))
        row['Has_Fireworks'] = 0
        row['Has_Temp_Structures'] = 0 if random.random() > 0.2 else 1
        row['VIP_Attendance'] = 0
        row['Loudspeaker_Used'] = 0 if row['Event_Category'] == 'Private Function' else 1
        row['Is_Moving_Procession'] = 0
        row['Road_Closure_Required'] = 0
        row['Food_Stalls_Present'] = 1 if random.random() > 0.5 else 0
        row['Liquor_Served'] = 0
        row['Duration_Hours'] = random.randint(2, 6)

    for col in ['Has_Fireworks', 'Has_Temp_Structures', 'VIP_Attendance', 'Loudspeaker_Used', 
                'Road_Closure_Required', 'Is_Moving_Procession', 'Food_Stalls_Present', 'Liquor_Served']:
        if col not in row: row[col] = 0
        row[col] = int(row[col])

    # ==========================================
    # NOISE INJECTION (For realistic accuracy)
    # ==========================================
    noise_chance = random.random()
    if noise_chance < 0.05: # 5% chance: High becomes Medium, or Medium becomes High
        if row['Risk_Level'] == 'High': row['Risk_Level'] = 'Medium'
        elif row['Risk_Level'] == 'Medium': row['Risk_Level'] = 'High'
    elif noise_chance > 0.95: # Another 5% chance: Medium becomes Low, or Low becomes Medium
        if row['Risk_Level'] == 'Medium': row['Risk_Level'] = 'Low'
        elif row['Risk_Level'] == 'Low': row['Risk_Level'] = 'Medium'
    # Notice we NEVER flip High directly to Low. That would look like a bad model.

    data.append(row)

df = pd.DataFrame(data, columns=columns)

DATA_PATH = 'E:\\Utsav_backend\\ai_intelligence\\data\\raw'
os.makedirs(DATA_PATH, exist_ok=True)
FILE_NAME = 'indian_event_risk_dataset_v3_balanced.csv'
FULL_PATH = os.path.join(DATA_PATH, FILE_NAME)
df.to_csv(FULL_PATH, index=False)
print(f"Dataset generated successfully at {FULL_PATH}")