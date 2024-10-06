import numpy as np 
import pandas as pd


rows = 54
np.random.seed(42)

gender = np.random.choice(['Male', 'Female'], rows,p=[0.6,0.4])

probabilities =[0.5]*2+[0.2]*3+ [0.1] *2+[0.08]*2  + [0.009] * 11  # Probabilities adjusted to favor values < 10
prob_sum = sum(probabilities)
probabilities = [p / prob_sum for p in probabilities]  # Normalize to sum to 1

years_of_experience = np.random.choice(range(1, 21), rows, p=probabilities)

teamwork_frequency = []
for g, y in zip(gender, years_of_experience):
    if y >= 10:  # High years of experience, increase probability for 'Usually'
        teamwork_freq = np.random.choice(['Never', 'Sometimes', 'Usually'], p=[0.005, 0.1, 0.895])
    elif y >= 7:  # Medium years of experience, balanced probability
        teamwork_freq = np.random.choice(['Never', 'Sometimes', 'Usually'], p=[0.05, 0.2, 0.75])
    else:  # Low years of experience, probability depends on gender
        if g == 'Male':
            teamwork_freq = np.random.choice(['Never', 'Sometimes', 'Usually'], p=[0.2, 0.4, 0.4])
        else:
            teamwork_freq = np.random.choice(['Never', 'Sometimes', 'Usually'], p=[0.1, 0.3, 0.6])
    
    teamwork_frequency.append(teamwork_freq)

# Creating the DataFrame
df = pd.DataFrame({
    'Gender': gender,
    'Years of Experience': years_of_experience,
    'How often do you work in teams?': teamwork_frequency
})



ea_columns = ['EA1', 'EA2', 'EA3', 'EA4', 'EA5', 'EA_level']
eu_columns = ['EU1', 'EU2', 'EU3', 'EU4', 'EU5', 'EU_level']
eus_columns = ['EUS1', 'EUS2', 'EUS3', 'EUS4', 'EUS5', 'EUS_level']
ec_columns = ['EC1', 'EC2', 'EC3', 'EC_level']
r_columns = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R_level']

def fill_values_with_level(level, num_values):
    filled_values = []
    for _ in range(num_values):
        if level == 'low':
            value = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])  # Low range distribution
        elif level == 'medium':
            value = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])  # Medium range distribution
        else:  # High
            value = np.random.choice([3,4, 5], p=[0.1,0.4, 0.5])  # High range distribution
        filled_values.append(value)
    filled_values.append(level)
    return filled_values

def fill_category_values(df, columns):
    # Iterate over each row in the DataFrame
    for row in range(len(df)):
        # Randomly select a level for this row
        level = np.random.choice(['low', 'medium', 'high'], p=[0.2, 0.4, 0.4])
        # Get the number of values to be filled (excluding the level column)
        num_values = len(columns) - 1
        # Get the values to fill the row from the fill_values_with_level function
        values = fill_values_with_level(level, num_values)
        # Fill the columns for this row
        df.loc[row, columns] = values
def fill_category_values_with_given_level(df, columns, level_column,corr):
    # Iterate over each row in the DataFrame
    for row in range(len(df)):
        # Get the specific level from the level column for the current row
        given_level = df.loc[row, level_column]
        
        if corr=='high':
            if given_level == 'high':
                probability=[0.05,0.05,0.9]
            elif given_level == 'medium':
                probability=[0.05,0.9,0.05]
            else:
                probability=[0.85,0.1,0.05]
        elif corr=='medium':
            if given_level == 'high':
                probability=[0.1,0.2,0.7]
            elif given_level == 'medium':
                probability=[0.1,0.7,0.2]
            else:
                probability=[0.6,0.3,0.1]
        else:
            if given_level == 'high':
                probability=[0.2,0.3,0.5]
            elif given_level == 'medium':
                probability=[0.2,0.5,0.3]
            else:
                probability=[0.5,0.3,0.2]

        level = np.random.choice(['low', 'medium', 'high'], p=probability)
        
        # Get the number of values to be filled (excluding the level column)
        num_values = len(columns) - 1
        # Get the values to fill the row from the fill_values_with_level function
        values = fill_values_with_level(level, num_values)
        # Fill the columns for this row
        df.loc[row, columns] = values


fill_category_values(df, r_columns)

fill_category_values_with_given_level(df, eu_columns, 'R_level','high')
fill_category_values_with_given_level(df, ec_columns, 'R_level','high')
fill_category_values_with_given_level(df, ea_columns, 'R_level','medium')
fill_category_values_with_given_level(df, eus_columns, 'R_level','low')
print(df.head())
df.to_csv('D:/vs code/Misc/data.csv')