import csv
import numpy as np
import collections
import pickle
import os

'''CONSTANTS'''
DEMO_FILE = '../data/demographic 5_5_15.txt'
EGM_FILE = '../data/EGM_preprocessed.txt'
METH_FILE = '../data/EGM-output online tool.csv'
PICKLE_FILE = '../cache/data.pkl'

'''load demographics data'''
def load_demos(demo_file):
    demo_profiles = {}
    with open(demo_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            if len(row) != 0:
                demo_profiles[row[0]] = row[1:]
    demo_profiles.pop("variable", None) #removes header row

    return demo_profiles

'''load EGM data'''
def load_EGM(egm_file):
    egm_profiles = collections.defaultdict(list)
    with open(egm_file) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t')
        for row in csv_reader:
            for (k,v) in row.items():
                egm_profiles[k[5:]].append(v) #appends under patient w/o "data."
    dict(egm_profiles)
    egm_profiles.pop("", None) #removes first column

    return egm_profiles

'''weed out patients not in both datasets'''
def intersect_dicts(dict1, dict2):
    for key in list(dict1):
        if not key in dict2:
            dict1.pop(key, None)
    for key in list(dict2):
        if not key in dict1:
            dict2.pop(key, None)

'''weed out patients with BRCA and return one-hot for cancer'''
def filter_cancer(demo_profiles, egm_profiles):
    cancer = {}
    for patient in list(demo_profiles):
        profile = demo_profiles[patient]
        #filter out BRCA
        if profile[10] == "BRCA":
            demo_profiles.pop(patient, None)
            egm_profiles.pop(patient, None)
        #build one_hot dict
        elif profile[10] == "CO":
            cancer[patient] = 0
        else:
            cancer[patient] = 1
    return cancer

def dict_to_numpy(d):
    dlist = list(d.items())
    dlist.sort()
    l1, l2 = zip(*dlist)
    l1_np = np.array(l1)
    l2_np = np.array(l2,dtype=float)
    
    return l1_np, l2_np

def load_BioAge1HO():
    profiles = {}

    #read data file
    with open(METH_FILE) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first_row = True
        #columns to exclude
        excluded_columns = [2,3,5,6]
        excluded_columns.sort(reverse=True)
        for row in csv_reader:
            if first_row:
                first_row = False
            else:
                patient = row[0]
                #excluding unwanted columns
                for i in excluded_columns:
                    del row[i]
                #storing patient profiles
                profiles[patient] = row[1:]
    

def get_data():
    if os.path.exists(PICKLE_FILE):
        #return pickled data
        pkl_file = open(PICKLE_FILE, 'rb')
        return pickle.load(pkl_file)
    else:
        #process data
        demo_profiles = load_demos(DEMO_FILE)
        egm_profiles = load_EGM(EGM_FILE)
        intersect_dicts(demo_profiles, egm_profiles)
        cancer_dict = filter_cancer(demo_profiles, egm_profiles)
        patients, egm_matrix =  dict_to_numpy(egm_profiles)
        patients, cancer_onehot = dict_to_numpy(cancer_dict)
        #pickle and return
        pkl_file = open(PICKLE_FILE, 'wb')
        pickle.dump([patients, egm_matrix, cancer_onehot],pkl_file)
        return patients, egm_matrix, cancer_onehot

def  main():
    patients, egm_matrix, cancer_onehot = get_data()



if __name__ == '__main__':
    main()

