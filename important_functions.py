#%%
# IMPORT FUNCTIONS
import myokit
import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
from scipy import stats
import pickle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.pyplot as plt
import time 
from multiprocessing import Pool
import myokit.formats

# DEFINE FUNCTIONS
def get_ind(vals = [1,1,1,1,1,1,1,1,1,1], celltype = 'adult'):
    if celltype == 'ipsc':
        tunable_parameters=['i_cal_pca_multiplier','i_ks_multiplier','i_kr_multiplier','i_na_multiplier','i_to_multiplier','i_k1_multiplier','i_f_multiplier']
        ind = dict(zip(tunable_parameters, vals))
    else:
        tunable_parameters=['i_cal_pca_multiplier','i_ks_multiplier','i_kr_multiplier','i_nal_multiplier','i_na_multiplier','i_to_multiplier','i_k1_multiplier','i_NCX_multiplier','i_nak_multiplier','i_kb_multiplier']
        ind = dict(zip(tunable_parameters, vals))
    return(ind)

def run_model(ind, beats, stim = 5.3, stim_1 = 0, start = 0.1, start_1 = 0, length = 1, length_1 = 0, cl = 1000, prepace = 600, I0 = 0, path = './models/', model = 'tor_ord_endo2.mmt'): 
    mod, proto = get_ind_data(ind, path, model = model)
    proto.schedule(stim, start, length, cl, 0) 
    if stim_1 != 0:
        proto.schedule(stim_1, start_1, length_1, cl, 1)
    sim = myokit.Simulation(mod,proto)

    if I0 != 0:
        sim.set_state(I0)

    sim.pre(cl * prepace) #pre-pace for 100 beats
    dat = sim.run(beats*cl) 
    IC = sim.state()

    return(dat, IC) 

def rrc_search(ind, IC, stim = 5.3, start = 0.2, length = 1, cl = 1000, path = './models/', model = 'tor_ord_endo2.mmt'):
    all_data = []
    APs = list(range((10*cl)+int(start+length+4), (100*cl)+int(start+length+4), 5*cl)) #length needs to be an integer so it rounds if needed

    mod, proto = get_ind_data(ind, path, model) 
    proto.schedule(stim, start, length, cl, 0)
    proto.schedule(0.3, (5*cl)+int(start+length+4), cl-int(start+length+4), cl, 1)
    sim = myokit.Simulation(mod, proto)
    sim.set_state(IC)
    dat = sim.run(7*cl)

    d0 = get_last_ap(dat, 4, cl=cl)
    result_abnormal0 = detect_abnormal_ap(d0['t'], d0['v']) 
    all_data.append({**{'t_rrc': d0['t'], 'v_rrc': d0['v'], 'stim': 0}, **result_abnormal0})

    d3 = get_last_ap(dat, 5, cl=cl)
    result_abnormal3 = detect_abnormal_ap(d3['t'], d3['v'])
    all_data.append({**{'t_rrc': d3['t'], 'v_rrc': d3['v'], 'stim': 3}, **result_abnormal3})

    #if result_EAD0 == 1 or result_RF0 == 1:
    if result_abnormal0['result'] == 1:
        RRC = 0

    #elif result_EAD3 == 0 and result_RF3 == 0:
    elif result_abnormal3['result'] == 0:
        # no abnormality at 0.3 stim, return RRC
        RRC = 0.3

    else:
        low = 0
        high = 0.3
        for i in list(range(0,len(APs))):
            mid = round((low + (high-low)/2), 4) 

            sim.reset()
            sim.set_state(IC)
            proto.schedule(mid, APs[i], cl-int(start+length+4), cl, 1)
            sim.set_protocol(proto)
            dat = sim.run(APs[i]+(2*cl))

            data = get_last_ap(dat, int((APs[i]-int(start+length+4))/cl), cl=cl)
            result_abnormal = detect_abnormal_ap(data['t'], data['v'])
            all_data.append({**{'t_rrc': data['t'], 'v_rrc': data['v'], 'stim': mid}, **result_abnormal})
            
            if result_abnormal['result'] == 0:
                # no RA so go from mid to high
                low = mid

            else:
                #repolarization failure so go from mid to low 
                high = mid

            #if (high-low)<0.01:
            if (high-low)<0.0025: #THIS WAS USED IN GA 8 AND BEFORE
                break 
        
        for i in list(range(1, len(all_data))):
            if all_data[-i]['result'] == 0:
                RRC = all_data[-i]['stim']
                break
            else:
                RRC = 0 #in this case there would be no stim without an RA

    result = {'RRC':RRC, 'data':all_data}

    return(result)

def get_last_ap(dat, AP, cl = 1000, type = 'full'):

    if type == 'full':
        start_ap = list(dat['engine.time']).index(closest(list(dat['engine.time']), AP*cl))
        end_ap = list(dat['engine.time']).index(closest(list(dat['engine.time']), (AP+1)*cl))

        t = np.array(dat['engine.time'][start_ap:end_ap])
        t = t-t[0]
        v = np.array(dat['membrane.v'][start_ap:end_ap])
        cai = np.array(dat['intracellular_ions.cai'][start_ap:end_ap])
        i_ion = np.array(dat['membrane.i_ion'][start_ap:end_ap])
        i_stim = np.array(dat['stimulus.i_stim'][start_ap:end_ap])

        data = {}
        data['t'] = t
        data['v'] = v
        data['cai'] = cai
        data['i_ion'] = i_ion
        data['i_stim'] = i_stim
    
    else:
        # Get t, v, and cai for second to last AP#######################
        ti, vol = dat

        start_ap = list(ti).index(closest(ti, AP*cl))
        end_ap = list(ti).index(closest(ti, (AP+1)*cl))

        t = np.array(ti[start_ap:end_ap])
        v = np.array(vol[start_ap:end_ap])

        data = {}
        data['t'] = t
        data['v'] = v

    return (data)

def check_physio(ap_features, feature_targets = {'Vm_peak': [10, 33, 55], 'dvdt_max': [100, 347, 1000], 'cat_amp': [3E-4*1e5, 3.12E-4*1e5, 8E-4*1e5], 'cat_peak': [40, 58, 60], 'cat90': [350, 467, 500]}):
    # C2 Calculation
    error = 0
    for k, v in feature_targets.items():
        if ((ap_features[k] > v[0]) and (ap_features[k] < v[2])):
            error+=0
        else:
            error+=(v[1]-ap_features[k])**2

    return(error)

def get_rrc_error(RRC):
    # C3 Calculation
    #################### RRC DETECTION & ERROR CALCULATION ##########################
    error = round((0.3 - (np.abs(RRC)))*20000)

    return error

def get_features(t,v,cai=None):

    ap_features = {}

    # Returns really large error value if cell AP is not valid 
    if ((min(v) > -60) or (max(v) < 0)):
        return 50000000 

    # Voltage/APD features#######################
    mdp = min(v)
    max_p = max(v)
    dvdt_max = np.max(np.diff(v[0:100])/np.diff(t[0:100]))

    ap_features['Vm_peak'] = max_p
    ap_features['dvdt_max'] = dvdt_max

    for apd_pct in [40, 50, 90]:
        apd_val = calc_APD(t,v,apd_pct) 
        ap_features[f'apd{apd_pct}'] = apd_val
 
    ap_features['triangulation'] = ap_features['apd90'] - ap_features['apd40']
    ap_features['RMP'] = np.mean(v[len(v)-50:len(v)])

    if cai is not None: 
        # Calcium/CaT features######################## 
        max_cai = np.max(cai)
        max_cai_idx = np.argmax(cai)
        max_cai_time = t[max_cai_idx]
        cat_amp = np.max(cai) - np.min(cai)
        ap_features['cat_amp'] = cat_amp * 1e5 #added in multiplier since number is so small
        ap_features['cat_peak'] = max_cai_time

        for cat_pct in [90]:
            cat_recov = max_cai - cat_amp * cat_pct / 100
            idx_catd = np.argmin(np.abs(cai[max_cai_idx:] - cat_recov))
            catd_val = t[idx_catd+max_cai_idx]

            ap_features[f'cat{cat_pct}'] = catd_val 

    return ap_features

def check_physio_torord(t, v, filter = 'no', path = './data'):
    # C3 Calculation
    # Cut off the upstroke of the AP for profile
    t_ind = list(t[150:len(t)]) 
    v_ind = list(v[150:len(t)])

    # Baseline tor-ord model & cut off upstroke
    base_df = pd.read_csv('./data/baseline_torord_data.csv.bz2')
    t_base = list(base_df['t'])[150:len(t)]
    v_base = list(base_df['v'])[150:len(t)]

    # Cut off the upstroke of the AP for the tor-ord data
    if filter == 'no':
        time, vol_10, vol_90 = get_torord_phys_data(path)
        t = time[150:len(time)]
        v_10 = vol_10[150:len(time)]
        v_90 = vol_90[150:len(time)]

    else:
        t, v_10, v_90 = get_torord_phys_data(path)

    result = 0 # valid AP
    error = 0
    check_times = []
    data = {}

    for i in list(range(0, len(t_ind))):
        t_dat = closest(t, t_ind[i]) # find the value closest to the ind's time within the exp data time list
        t_dat_base = closest(t_base, t_ind[i])
        t_dat_i = np.where(np.array(t)==t_dat)[0][0] #find the index of the closest value in the list 
        t_dat_base_i = np.where(np.array(t_base)==t_dat_base)[0][0] #find the index of the closest value in the list 
        v_model = v_ind[i]
        v_lowerbound = v_10[t_dat_i]
        v_upperbound = v_90[t_dat_i]
        v_torord = v_base[t_dat_base_i] 

        check_times.append(np.abs(t_ind[i] - t_dat))

        if v_model < v_lowerbound or v_model > v_upperbound:
            result = 1 # not a valid AP
            error += (v_model - v_torord)**2
    
    data['result'] = result
    data['error'] = error
    data['check_times'] = check_times

    return(data)

def get_torord_phys_data(path):
    data = pd.read_csv(path+'/APbounds.csv.bz2')
    time = [x - 9.1666666669999994 for x in list(data['t'])] #shift action potential to match solutions
    t = time[275:len(data['v_10'])]
    v_10 = list(data['v_10'])[275:len(data['v_10'])]
    v_90 = list(data['v_90'])[275:len(data['v_10'])]

    data = pd.DataFrame(data = {'t': t[1000:len(t)], 'v_10': v_10[1000:len(t)], 'v_90':v_90[1000:len(t)]})
    data_start = pd.DataFrame(data = {'t': t[150:1000], 'v_10': v_10[150:1000], 'v_90':v_90[150:1000]})
    
    # FILTER V_10
    v_10_new = data.v_10.rolling(400, min_periods = 1, center = True).mean()
    v_10_start = data_start.v_10.rolling(100, min_periods = 1, center = True).mean()
    v_10_new = v_10_new.dropna()
    v_10 = list(v_10_start) + list(v_10_new)
    t = list(data_start['t']) + list(data['t'])

    # FILTER V_90
    v_90_new = data.v_90.rolling(400, min_periods = 1, center = True).mean()
    v_90_start = data_start.v_90.rolling(200, min_periods = 1, center = True).mean()
    v_90_new = v_90_new.dropna()
    v_90 = list(v_90_start) + list(v_90_new)


    return(t, v_10, v_90)

def calc_APD(t, v, apd_pct):
    t = np.array(t)
    v = np.array(v)
    t = [i-t[0] for i in t]
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    repol_pot = max_p - apa * apd_pct/100
    idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
    apd_val = t[idx_apd+max_p_idx]

    result = detect_abnormal_ap(t, v)
    if len(result['RMP']) == 0:
        apd_val = max(t)

    return(apd_val) 

def get_ind_data(ind, path = './models/', model = 'tor_ord_endo2.mmt'):
    mod, proto, x = myokit.load(path+model)
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    return mod, proto

def detect_abnormal_ap(t, v):

    slopes = []
    for i in list(range(0, len(v)-1)):
        if t[i] > 100 and v[i] < 20:
            m = (v[i+1]-v[i])/(t[i+1]-t[i])
            slopes.append(round(m, 2))
        else:
            slopes.append(-2.0)

    # EAD CODE
    rises_idx = np.where(np.array(slopes)>0)
    rises_groups = []
    for k, g in groupby(enumerate(rises_idx[0]), lambda i_x: i_x[0] - i_x[1]):
        rises_groups.append(list(map(itemgetter(1), g)))

    # RF CODE
    rpm_idx = np.where(np.array(slopes) == 0)
    rpm_groups = []
    for k, g in groupby(enumerate(rpm_idx[0]), lambda i_x: i_x[0] - i_x[1]):
        rpm_groups.append(list(map(itemgetter(1), g)))

    flat_groups = [group for group in rpm_groups if v[group[-1]]<-70]

    # CHECK PHASE 4 RF
    if len(flat_groups)>0:
        RMP_start = flat_groups[0][0]
        v_rm = v[RMP_start:len(v)]
        t_rm = t[RMP_start:len(v)]
        slope = (v_rm[-1]-v_rm[0])/(t_rm[-1]-t_rm[0])
        if slope < 0.01:
            for group in rises_groups:
                if v[group[0]]<-70:
                    rises_groups.remove(group)


    if len(flat_groups)>0 and len(rises_groups)==0:
        info = "normal AP" 
        result = 0
    else:
        info = "abnormal AP"
        result = 1

    data = {'info': info, 'result':result, 'EADs':rises_groups, 'RMP':flat_groups}

    return(data)

def closest(lst, K): 
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def add_scalebar(axs, section, y_pos = -0.1):
    # FORMAT X AXIS
    if section == 0:
        xmin, xmax, ymin, ymax = axs.axis()
        scalebar = AnchoredSizeBar(axs.transData, 100, '100 ms', 'lower left', bbox_to_anchor = (0,y_pos), bbox_transform =axs.transAxes, pad=0.5, color='black', frameon=False, size_vertical=(ymax-ymin)*0.0001) #fontproperties=fontprops
        axs.add_artist(scalebar)
        axs.spines[['bottom']].set_visible(False)
        axs.tick_params(bottom=False)
        axs.tick_params(labelbottom=False)
    else:
        for i in list(range(0, len(section))):
            xmin, xmax, ymin, ymax = axs[section[i][0], section[i][1]].axis()
            scalebar = AnchoredSizeBar(axs[section[i][0], section[i][1]].transData, 100, '100 ms', 'lower left', bbox_to_anchor = (0,y_pos), bbox_transform =axs[section[i][0], section[i][1]].transAxes, pad=0.5, color='black', frameon=False, size_vertical=(ymax-ymin)*0.0001) #fontproperties=fontprops
            axs[section[i][0], section[i][1]].add_artist(scalebar)
            axs[section[i][0], section[i][1]].spines[['bottom']].set_visible(False)
            axs[section[i][0], section[i][1]].tick_params(bottom=False)
            axs[section[i][0], section[i][1]].tick_params(labelbottom=False)

def new_parameter_convergence(all_trials, fitness='fitness'):
    all_dicts = []

    
    for t in list(range(0, max(all_trials['trial']))):
        old_data = all_trials[(all_trials['trial']==t) & (all_trials['gen']==0)].sort_values(fitness).iloc[0:100]
        for g in list(range(0, max(all_trials[(all_trials['trial']==t)]['gen']))):
            data = all_trials[(all_trials['trial']==t) & (all_trials['gen']==g)].sort_values(fitness).iloc[0:100]
            data = pd.concat([old_data, data])
            data = data.drop_duplicates(subset=data.filter(like='multiplier').columns.to_list())
            data_var = data.sort_values(fitness).iloc[0:100].filter(like = 'multiplier').var().to_dict()
            data_var['generation'] = g
            data_var['trial'] = t
            all_dicts.append(data_var)
            old_data = data

    df_dicts = pd.DataFrame(all_dicts)

    average_dicts = []
    for g in list(range(0, max(df_dicts['generation']))):
        average_dicts.append(df_dicts[df_dicts['generation']==g].mean().to_dict())
    df_dicts_average = pd.DataFrame(average_dicts)
    
    return df_dicts_average

def get_sensitivities(all_trials, error):
    population = all_trials[all_trials['gen']==0].sort_values(error) 
    good_pop = population.iloc[0:160] #top 10% of the population
    bad_pop = population.iloc[160:len(population['gen'])] #remaining 90% of the population

    sensitivities = []
    pvalues = []
    for cond in good_pop.filter(like='multiplier').columns.to_list():
        stat, pvalue = stats.ks_2samp(good_pop[cond], bad_pop[cond])
        sensitivities.append(stat)
        pvalues.append(pvalue)
    return sensitivities, pvalues

def check_robustness_old(best_data, conductance, values, i_kb = 1):
    
    all_data = []

    for value in values:
        if i_kb != 1:
            conductance_label = conductance + ' and i_kb_multiplier'
        else:
            conductance_label = conductance

        # baseline torord model 
        dat, IC = run_model([{conductance: value, 'i_kb_multiplier': i_kb}], 1)
        t = dat['engine.time']
        v = dat['membrane.v']
        apd90 = calc_APD(t, v, 90)
        data = detect_abnormal_ap(t, v)
        result = data['result']
        labels = ['conductance', 'value', 't', 'v', 'apd90', 'result', 'type']
        all_data.append(dict(zip(labels, [conductance_label, value, t, v, apd90, result, 'torord'])))

        # best ind
        best_ind = best_data.filter(like = 'multiplier').iloc[0].to_dict()
        best_ind['i_kb_multiplier'] = i_kb
        best_ind[conductance] = best_ind[conductance]*value
        best_dat, best_IC = run_model([best_ind], 1)
        t_best = best_dat['engine.time']
        v_best = best_dat['membrane.v']
        apd90_best = calc_APD(t_best, v_best, 90)
        data_best = detect_abnormal_ap(t_best, v_best)
        result_best = data_best['result']
        all_data.append(dict(zip(labels, [conductance_label, value, t_best, v_best, apd90_best, result_best, 'optimized'])))

    return(all_data)

def check_robustness(args):
    
    i, best_data, conductance, values, i_kb = args
    all_data = {'type': 'optimized '+str(i)}

    if i_kb != 1:
        all_data['conductance'] = conductance + ' & i_kb_multiplier'
    else:
        all_data['conductance'] = conductance

    for value in values:

        # best ind
        best_ind = best_data.filter(like = 'multiplier').iloc[i].to_dict()
        best_ind['i_kb_multiplier'] = i_kb
        best_ind[conductance] = best_ind[conductance]*value

        try:
            best_dat, best_IC = run_model([best_ind], 5)
            t_best = best_dat['engine.time']
            v_best = best_dat['membrane.v']
            apd90s = []
            results = []
            for a in [0, 1, 2, 3, 4]:
                ap_data = get_last_ap([t_best, v_best], a, type = 'half')
                apd90s.append(calc_APD(ap_data['t'], ap_data['v'], 90))
                results.append(detect_abnormal_ap(ap_data['t'], ap_data['v'])['result'])

            #save data to existing dict
            all_data['t_'+str(value)] = str(list(t_best))
            all_data['v_'+str(value)] = str(list(v_best))
            all_data['apd90_'+str(value)] = str(apd90s)
            all_data['result_'+str(value)] = str(results)
        except:
            all_data['t_'+str(value)] = 2000
            all_data['v_'+str(value)] = 2000
            all_data['apd90_'+str(value)] = 2000
            all_data['result_'+str(value)] = 2000
    return(all_data)

def collect_rrc_data(args):
    i, best_conds, stims = args
    ind_data = {}
    ind = best_conds.filter(like= 'multiplier').iloc[i].to_dict()
    for s in list(range(0, len(stims))):
        dat, IC = run_model([ind], 1, stim = 5.3, stim_1 = stims[s], start = 0.1, start_1 = 4, length = 1, length_1 = 996, cl = 1000, prepace = 600, I0 = 0, path = './models/', model = 'tor_ord_endo2.mmt') 
        ind_data['t_'+str(stims[s])] = [dat['engine.time'].tolist()]
        ind_data['v_'+str(stims[s])] = [dat['membrane.v'].tolist()]
        ind_data['apd_'+str(stims[s])] = calc_APD(dat['engine.time'], dat['membrane.v'], 90)
        if s == 0:
            base_apd = calc_APD(dat['engine.time'], dat['membrane.v'], 90)
        ind_data['delapd_'+str(stims[s])] = calc_APD(dat['engine.time'], dat['membrane.v'], 90) - base_apd
        
    data = {**ind, **ind_data}
    return data

def collect_sensitivity_data(args):
    og_ind, i, cond, purturb, p, model, stim, length = args
    data = {}

    dat_initial, IC_initial = run_model([og_ind], 5, prepace = 600,  path = p, model = model, stim = stim, length=length)
    print('finished running model')
    
    results_initial = rrc_search([og_ind], IC_initial, path = p, model=model, stim=stim, length=length) 
    # Get RRC error - C3
    rrc_fitness = get_rrc_error(results_initial['RRC'])
    print('finished rrc search')

    apds = []
    features = []
    morphs = []
    for ap in list(range(0,5)):
        ap_data_initial = get_last_ap(dat_initial, ap, cl = 1000)
        features_initial = get_features(ap_data_initial['t'], ap_data_initial['v'], ap_data_initial['cai'])
        if features_initial == 50000000:
            features_initial = {'Vm_peak': 50000000, 'dvdt_max': 50000000, 'apd40': 50000000, 'apd50': 50000000, 'apd90': 50000000, 'triangulation': 50000000, 'RMP': 50000000, 'cat_amp': 50000000, 'cat_peak': 50000000, 'cat90': 50000000}

        # Get APD90
        initial_APD90 = calc_APD(ap_data_initial['t'], ap_data_initial['v'], 90)
        apds.append(initial_APD90)

        # Get feature error -C2
        initial_feature_error = check_physio(features_initial)
        features.append(initial_feature_error)

        # Get Morphology error -C1
        initial_morph_error = check_physio_torord(ap_data_initial['t'], ap_data_initial['v'], filter = 'yes')
        morphs.append(initial_morph_error['error']) 

    data['t_initial'] = list(dat_initial['engine.time'])
    data['v_initial'] = list(dat_initial['membrane.v'])
    data['rrc_error_initial'] = rrc_fitness
    data['rrc_initial'] = results_initial['RRC']
    data['apd_initial'] = apds
    data['feature_error_initial'] = sum(features)
    data['morph_error_initial'] = sum(morphs)

    # FINISHED INITIAL CONDITIONS
    print('finished initial conditions')


    for pur in list(range(0,len(purturb))):
        ind = og_ind.copy()
        ind[cond[i]] = ind[cond[i]]*purturb[pur] #purturbation as multiplier
        #ind[cond[i]] = purturb[pur] #purturbation as value
        print(cond[i], purturb[pur], ind)
        dat_long, IC_long = run_model([ind], 5, prepace = 600, I0 = IC_initial, path = p, model = model, stim=stim, length=length)
        

        results = rrc_search([ind], IC_long, path = p, model=model, stim=stim, length=length) 
        RRC_fitness = get_rrc_error(results['RRC'])

        apds = []
        features = []
        morphs = []
        for ap in list(range(0,5)):
            ap_data_final = get_last_ap(dat_long, ap, cl = 1000)
            features_final = get_features(ap_data_final['t'], ap_data_final['v'], ap_data_final['cai'])
            if features_final == 50000000:
                features_final = {'Vm_peak': 50000000, 'dvdt_max': 50000000, 'apd40': 50000000, 'apd50': 50000000, 'apd90': 50000000, 'triangulation': 50000000, 'RMP': 50000000, 'cat_amp': 50000000, 'cat_peak': 50000000, 'cat90': 50000000}

            # Get APD90
            final_APD90 = calc_APD(ap_data_final['t'], ap_data_final['v'], 90)
            apds.append(final_APD90)

            # Get feature error - C2
            final_feature_error = check_physio(features_final)
            features.append(final_feature_error)

            # Get Morphology error
            final_morph_error = check_physio_torord(ap_data_final['t'], ap_data_final['v'], filter = 'yes')
            morphs.append(final_morph_error['error']) 

        # SAVE DATA
        data['t_'+str(purturb[pur])] = list(dat_long['engine.time'])
        data['v_'+str(purturb[pur])] = list(dat_long['membrane.v'])
        data['rrc_error_'+str(purturb[pur])] = RRC_fitness
        data['rrc_'+str(purturb[pur])] = results['RRC']
        data['apd_'+str(purturb[pur])] = apds
        data['feature_error_'+str(purturb[pur])] = sum(features)
        data['morph_error_'+str(purturb[pur])] = sum(morphs)

        print('finished cond * ', purturb[pur])


    data = {**{'cond': cond[i]}, **data}

    return(data)

def get_baseline_torord_data(save_to = './data/baseline_torord_data.csv.bz2'):
    dat, IC = run_model(None, 1)
    baseline_torord = pd.DataFrame({'t': dat['engine.time'], 'v': dat['membrane.v']})
    baseline_torord.to_csv(save_to, index = False)

def get_cond_data(best_data_path = './data/best_data.csv.bz2', save_to = './data/cond_data.pkl'):
    """
    This function runs simulations in the ToR-ORd and Grandi models to generate action potential data. 
    Specically, this generates the data in the following manuscript figures: 4A, 5, 6B, 7, 8A, and 8B. 
    The data is stored as a pickle file contained in the data/ folder and named cond_data.pkl.
    """
    
    # LOAD DATA
    best_data = pd.read_csv(best_data_path)

    ##########################################################################################################################################################
    # RUN SIMULATIONS

    # Baseline Data - BM
    base_ind = get_ind()
    base_ind['i_bias_multiplier'] = 0
    base_ind['i_bias1_multiplier'] = 0
    dat, IC = run_model([base_ind], 1)
    result = rrc_search([base_ind], IC)
    all_data = {**base_ind, **result, **{'dat':dat}}

    # Optimized Data - OM
    ind = best_data.iloc[0].filter(like = 'multiplier').to_dict()
    ind['i_bias_multiplier'] = 0
    ind['i_bias1_multiplier'] = 0
    dat_o, IC_o = run_model([ind], 1)
    result_o = rrc_search([ind], IC_o)
    all_data_o = {**ind, **result_o, **{'dat':dat_o}}

    # Optimized Data without INaL - OM1
    ind_0 = ind.copy()
    ind_0['i_nal_multiplier'] = 0
    dat_0, IC_0 = run_model([ind_0], 1)
    result_0 = rrc_search([ind_0], IC_0)
    all_data_0 = {**ind_0, **result_0, **{'dat':dat_0}}

    # Optimized Data without INaL and ICaL Rescue - OM2
    ind_1 = ind_0.copy()
    ind_1['i_cal_pca_multiplier'] = 3
    dat_1, IC_1 = run_model([ind_1], 1)
    result_1 = rrc_search([ind_1], IC_1)
    all_data_1 = {**ind_1, **result_1, **{'dat':dat_1}}

    # Optimized Data without ICaL - OM3
    ind_0a = ind.copy()
    ind_0a['i_cal_pca_multiplier'] = 0
    dat_0a, IC_0a = run_model([ind_0a], 1)
    result_0a = rrc_search([ind_0a], IC_0a)
    all_data_0a = {**ind_0a, **result_0a, **{'dat':dat_0a}}

    # Baseline with fake outward - BM2
    ind_2 = base_ind.copy()
    ind_2['i_bias_multiplier'] = 0.8
    ind_2['i_bias1_multiplier'] = 0
    dat_2, IC_2 = run_model([ind_2], 1)
    result_2 = rrc_search([ind_2], IC_2)
    all_data_2 = {**ind_2, **result_2, **{'dat':dat_2}}

    # Baseline with fake outward and ICaL = 3 - BM3
    ind_3 = ind_2.copy()
    ind_3['i_cal_pca_multiplier'] = 3
    dat_3, IC_3 = run_model([ind_3], 1)
    result_3 = rrc_search([ind_3], IC_3)
    all_data_3 = {**ind_3, **result_3, **{'dat':dat_3}}

    # Baseline with fake outward and INaL = 3 - BM4,
    ind_4 = ind_2.copy()
    ind_4['i_nal_multiplier'] = 3
    dat_4, IC_4 = run_model([ind_4], 1)
    result_4 = rrc_search([ind_4], IC_4)
    all_data_4 = {**ind_4, **result_4, **{'dat':dat_4}}

    # Baseline with fake outward and fake inward - BM1
    ind_5 = ind_2.copy()
    ind_5['i_bias1_multiplier'] = -0.8
    dat_5, IC_5 = run_model([ind_5], 1)
    result_5 = rrc_search([ind_5], IC_5)
    all_data_5 = {**ind_5, **result_5, **{'dat':dat_5}}

    # Baseline Grandi Data - GBM
    base_ind_g = {'i_bias_multiplier': 0, 'i_bias1_multiplier': 0}
    dat_g, IC_g = run_model([base_ind_g], 1, model = 'grandi_flat.mmt', stim=1, length=5)
    result_g = rrc_search([base_ind_g], IC_g, model = 'grandi_flat.mmt', stim=1, length=5)
    all_data_g = {**base_ind_g, **result_g, **{'dat':dat_g}}

    # Baseline Grandi + Outward - GBM2
    base_ind_g2 = base_ind_g.copy()
    base_ind_g2['i_bias_multiplier'] = 0.8
    base_ind_g2['i_bias1_multiplier'] = 0
    dat_g2, IC_g2 = run_model([base_ind_g2], 1, model = 'grandi_flat.mmt', stim=1, length=5)
    result_g2 = rrc_search([base_ind_g2], IC_g2, model = 'grandi_flat.mmt', stim=1, length=5)
    all_data_g2 = {**base_ind_g2, **result_g2, **{'dat':dat_g2}}

    # Baseline Grandi + Outward + ICaL = 3 - GBM3
    base_ind_g3 = base_ind_g2.copy()
    base_ind_g3['i_cal_pca_multiplier'] = 3
    dat_g3, IC_g3 = run_model([base_ind_g3], 1, model = 'grandi_flat.mmt', stim=1, length=5)
    result_g3 = rrc_search([base_ind_g3], IC_g3, model = 'grandi_flat.mmt', stim=1, length=5)
    all_data_g3 = {**base_ind_g3, **result_g3, **{'dat':dat_g3}}

    # Baseline Grandi with fake outward and fake inward - GBM1
    base_ind_g5  = base_ind_g2.copy()
    base_ind_g5['i_bias1_multiplier'] = -0.8
    dat_g5, IC_g5 = run_model([base_ind_g5], 1, model = 'grandi_flat.mmt', stim=1, length=5)
    result_g5 = rrc_search([base_ind_g5], IC_g5, model = 'grandi_flat.mmt', stim=1, length=5)
    all_data_g5 = {**base_ind_g5, **result_g5, **{'dat':dat_g5}}

    # Baseline Grandi + Outward + INaL = 1 - GBM4
    base_ind_g4 = base_ind_g2.copy()
    base_ind_g4['i_nal_multiplier'] = 1
    dat_g4, IC_g4 = run_model([base_ind_g4], 1, model = 'grandi_flat.mmt', stim=1, length=5)
    result_g4 = rrc_search([base_ind_g4], IC_g4, model = 'grandi_flat.mmt', stim=1, length=5)
    all_data_g4 = {**base_ind_g4, **result_g4, **{'dat':dat_g4}}

    ##########################################################################################################################################################
    # SAVE DATA
    pickle.dump({'BM':all_data, 'OM':all_data_o, 'OM1':all_data_0, 'OM2':all_data_1, 'OM3':all_data_0a, 'BM2':all_data_2, 'BM3':all_data_3, 'BM4':all_data_4, 'BM1':all_data_5, 'GBM':all_data_g, 'GBM2':all_data_g2, 'GBM3':all_data_g3, 'GBM4':all_data_g4, 'GBM1':all_data_g5}, open(save_to, 'wb'))

def get_robust_data_old(best_data_path = './data/best_data.csv.bz2', save_to = './data/robust_data.pkl'):
    best_data = pd.read_csv(best_data_path)
    ical_data = check_robustness_old(best_data, 'i_cal_pca_multiplier', [1, 2, 3, 4, 5, 6, 7, 8])
    ikr_kb_data = check_robustness_old(best_data, 'i_kr_multiplier', [1, 0.8, 0.6, 0.4, 0.2, 0], i_kb = 0.6)
    ical_data.extend(ikr_kb_data)
    ikr_data = check_robustness_old(best_data, 'i_kr_multiplier', [1, 0.8, 0.6, 0.4, 0.2, 0])
    ical_data.extend(ikr_data)
    robust_df = pd.DataFrame(ical_data)

    pickle.dump(robust_df, open(save_to, 'wb'))

def get_robust_data(best_data_path = './data/best_data.csv.bz2', save_to = './data/robust_data_ical.csv.bz2', conductance = 'i_cal_pca_multiplier', values = [1, 2, 3, 4, 5, 6, 7, 8], i_kb = 1, multiprocessing = 'no'):
    """
    This function runs simulations to calculate the change in action potential duraction between the baseline ToR-ORd model and the 220 best GA individuals 
    at difference conductance purturbations. 
    
    """
    # LOAD DATA
    best_data = pd.read_csv(best_data_path) 
    best_conds = best_data.filter(like = 'multiplier')
    best_conds.loc[len(best_conds)] = [1]*9 # add baseline as last row
    #best_conds = best_conds.sort_index().reset_index(drop=True)

    ##########################################################################################################################################################
    # RUN SIMULATION
    print(time.time())
    time1 = time.time()

    #if __name__ == "__main__":

    index = len(best_conds['i_cal_pca_multiplier'])
    args = [(i, best_conds, conductance, values, i_kb) for i in range(index)]

    if multiprocessing == 'yes':
        p = Pool() #allocates for the maximum amount of processers on laptop
        result = p.map(check_robustness, args) 
        p.close()
        p.join()
    else:
        result = list(map(check_robustness, args))

    time2 = time.time()
    print('processing time: ', (time2-time1)/60, ' Minutes')
    print(time.time())

    ##########################################################################################################################################################
    # SAVE DATA
    df_data = pd.DataFrame(result)

    ##########################################################################################################################################################
    # Collect baseline Data
    all_data = {'type': 'baseline', 'conductance':conductance} 

    for value in values:

        # baseline torord model 
        dat, IC = run_model([{conductance: value, 'i_kb_multiplier': i_kb}], 5)
        t = dat['engine.time']
        v = dat['membrane.v']
        apd90s = []
        results = []
        for a in [0, 1, 2, 3, 4]:
            ap_data = get_last_ap([t, v], a, type = 'half')
            apd90s.append(calc_APD(ap_data['t'], ap_data['v'], 90))
            results.append(detect_abnormal_ap(ap_data['t'], ap_data['v'])['result'])
        all_data['t_'+str(value)] = str(list(t))
        all_data['v_'+str(value)] = str(list(v))
        all_data['apd90_'+str(value)] = str(apd90s)
        all_data['result_'+str(value)] = str(results)

    df_data.loc[len(df_data.index)] = list(all_data.values())

    ##########################################################################################################################################################
    # SAVE DATA
    df_data.to_csv(save_to, index = False)

def get_rrc_data(best_data_path = './data/best_data.csv.bz2', save_to = './data/rrc_data.csv.bz2', multiprocessing = 'no'):
    """
    This function runs simulations to calculate the change in action potential duraction between the baseline ToR-ORd model and the 220 best GA individuals 
    at various stimuli. This data was used to produce Figures 3E and 3F. The data is stored as a csv file contained in the data/ folder and named rrc_data.csv.bz2.
    
    """
    # LOAD DATA
    best_data = pd.read_csv(best_data_path) 
    best_conds = best_data.filter(like = 'multiplier')
    best_conds.loc[len(best_conds)-1] = [1]*9 # add baseline as last row
    best_conds = best_conds.sort_index().reset_index(drop=True)

    ##########################################################################################################################################################
    # RUN SIMULATION
    print(time.time())
    time1 = time.time()

    #if __name__ == "__main__":

    index = len(best_conds['i_cal_pca_multiplier'])
    args = [(i, best_conds, [0, 0.05, 0.1, 0.15, 0.2]) for i in range(index)]

    if multiprocessing == 'yes':
        p = Pool() #allocates for the maximum amount of processers on laptop
        result = p.map(collect_rrc_data, args) 
        p.close()
        p.join()
    else:
        result = list(map(collect_rrc_data, args))

    time2 = time.time()
    print('processing time: ', (time2-time1)/60, ' Minutes')
    print(time.time())

    ##########################################################################################################################################################
    # SAVE DATA
    df_data = pd.DataFrame(result)
    df_data.to_csv(save_to, index = False)

def get_local_sensitivity(best_data_path = './data/best_data.csv.bz2', save_to = './data/apd_sens_opt.csv.bz2', model = 'tor_ord_endo2.mmt', stim = 5.3, length =1, multiprocessing = 'no'):
    # LOAD DATA
    

    # If there is a path given to best data then run code with a representitive optimized model. If not, run baseline.  
    if isinstance(best_data_path, str):
        best_data = pd.read_csv(best_data_path) 
        ind = best_data.filter(like = 'multiplier').iloc[0].to_dict() #optimized mode
    else:
        ind = best_data_path


    ##########################################################################################################################################################
    # RUN SIMULATION
    print(time.time())
    time1 = time.time()

    #if __name__ == "__main__":

    index = 9  # this is the length of the population
    args = [(ind, i, ['i_cal_pca_multiplier', 'i_ks_multiplier', 'i_kr_multiplier', 'i_nal_multiplier', 'i_na_multiplier', 'i_to_multiplier', 'i_k1_multiplier', 'i_NCX_multiplier', 'i_nak_multiplier'], [0.3, 1, 1.5, 2, 2.5, 3], './models/', model, stim, length) for i in range(index)]

    if multiprocessing == 'yes':
        p = Pool() #allocates for the maximum amount of processers on laptop
        result = p.map(collect_sensitivity_data, args) 
        p.close()
        p.join()
    else:
        result = list(map(collect_sensitivity_data, args))

    time2 = time.time()
    print('processing time: ', (time2-time1)/60, ' Minutes')
    print(time.time())

    ##########################################################################################################################################################
    # SAVE DATA
    df_data = pd.DataFrame(result)
    df_data.to_csv(save_to, index = False)

def generate_alldata(get_raw_data, trials = ['trial1', 'trial2', 'trial4', 'trial5', 'trial6', 'trial8', 'trial9', 'trial10']):

        # First, combine all the raw data for each individual trial into one dataframe
        all_trials = pd.read_csv(get_raw_data+trials[0]+'_info.csv.bz2')  
        all_trials['trial'] = 0

        for t in list(range(1, len(trials))):
            print(trials[t])
            data = pd.read_csv(get_raw_data+trials[t]+'_info.csv.bz2') 
            data['trial'] = t
            all_trials = all_trials.append(data)

        all_trials = all_trials.drop_duplicates(subset=all_trials.filter(like='multiplier').columns.to_list()) #drop dublicates so all best individuals are unique 
        
        # Second, add an error for the apd 90 of the last action potential of each individual. 
        # The lower and upper bounds are based on values reported in the literature (using the same papers as the other biomarkers in the manuscript)

        """
        apds = all_trials['apd90_AP4'].tolist()
        apd_error = []
        lower_bound = 180
        upper_bound = 440
        mid_point = 180+(440-180)/2
        for apd in apds:
            if apd > lower_bound and apd < upper_bound:
                apd_error.append(0)
            else:
                apd_error.append(abs(mid_point-apd))

        all_trials['apd90_AP4_error'] = apd_error
        """

        return(all_trials)

def cellml_to_mmt(model_cellml, model_mmt):
    i = myokit.formats.importer('cellml')
    mod=i.model(model_cellml) 
    myokit.save(model_mmt, mod)
# %%
