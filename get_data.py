#%%
# IMPORT FUNCTIONS
from important_functions import get_local_sensitivity, get_rrc_data, get_ind

save_data_to = './data/'
get_data_from = './data/'

# Sensitivity data for baseline Grandi Model
get_local_sensitivity(best_data_path={'i_cal_pca_multiplier':1, 'i_ks_multiplier':1, 'i_kr_multiplier':1, 'i_nal_multiplier':0, 'i_na_multiplier':1, 'i_to_multiplier':1, 'i_k1_multiplier':1, 'i_NCX_multiplier':1, 'i_nak_multiplier':1}, save_to = save_data_to+'sens_grandi.csv.bz2', model = 'grandi_flat.mmt', stim = 1, length = 5, multiprocessing='yes')

# Sensitivity data for optimized ToR-ORd Model
get_local_sensitivity(best_data_path=get_data_from+'best_data.csv.bz2', save_to = save_data_to+'sens_opt.csv.bz2', multiprocessing='yes')

# Sensitivity data for baseline ToR-ORd Model
get_local_sensitivity(best_data_path=get_ind(), save_to = save_data_to+'sens_baseline.csv.bz2', multiprocessing='yes')

# RRC data for baseline ToR-ORd Model and all best data models
get_rrc_data(best_data_path=get_data_from+'best_data.csv.bz2', save_to = save_data_to+'rrc_data.csv.bz2', multiprocessing='yes')