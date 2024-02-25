from dotenv import load_dotenv
import os
import pandas as pd
load_dotenv()


population_path = os.path.join('data', 'population.csv')

print(population_df = pd.read_csv(population_path))

