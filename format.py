import csv
import numpy
import collections

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

def intersect_dicts(dict1, dict2):
    for key,v in dict1.items():
        if not key in dict2:
            dict1.pop(key, None)
    for key,v in dict2.items():
        if not key in dict1:
            dict2.pop(key, None)

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

def main():
    demo_profiles = load_demos('data/demographic 5_5_15.txt')
    egm_profiles = load_EGM('data/EGM_preprocessed.txt')

    count = 0
    for patient in egm_profiles:
        if patient in demo_profiles:
            count +=1
    print len (demo_profiles)
    print len (egm_profiles)
    print count
    intersect_dicts(demo_profiles, egm_profiles)
    print len (demo_profiles)
    print len (egm_profiles)

if __name__ == '__main__':
    main()

