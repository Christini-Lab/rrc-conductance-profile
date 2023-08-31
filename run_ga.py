#%%
# IMPORT FUNCTIONS
import os
from supercell_ga import start_ga

def main():
    all_individuals = start_ga(pop_size = 5, max_generations=3, path_to_model='./models/', save_data_to=os.environ['save_name'], multithread='yes')
    return(all_individuals)

if __name__ == '__main__':
    all_individuals = main()

# %%
