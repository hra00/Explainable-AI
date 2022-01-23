import shutil, glob, os

def split_data(folderpath='./data/RAF-DB/'):
    gender_dict = {}
    race_dict = {}
    age_dict = {}
    for test in ['train', 'test']:
        for i in range(15000):
            number = str(i+1).rjust(5,'0') if test=='train' else str(i+1).rjust(4,'0')
            path = folderpath + 'Annotation/manual/'+test+'_'+number+'_manu_attri.txt'
            try:
                listt = open(path, 'r').read().strip().split('\n')
                gender, race, age_group = listt[5:]
            except: break
            gender_dict[test+'_'+number] = gender
            race_dict[test+'_'+number] = race
            age_dict[test+'_'+number] = age_group
    
    src_dir = folderpath + 'Image/aligned/'
    dst_dir = folderpath + "biased/"

    for key, value in gender_dict.items():
        img_dir = src_dir + key + '_aligned.jpg'
        if value == '0': 
            dist = dst_dir + "male/"
        else: 
            dist = dst_dir + "female/"
        dist += key + '.jpg'
        shutil.copy(img_dir, dist)

    for key, value in race_dict.items():
        img_dir = src_dir + key + '_aligned.jpg'
        if value == '0': 
            dist = dst_dir + "caucasian/"
        elif value == '1': 
            dist = dst_dir + "african-american/"
        else: 
            dist = dst_dir + "asian/"
        dist += key + '.jpg'
        shutil.copy(img_dir, dist)

    for key, value in age_dict.items():
        img_dir = src_dir + key + '_aligned.jpg'
        if value == '0': 
            dist = dst_dir + "0-3/"
        elif value == '1': 
            dist = dst_dir + "4-19/"
        elif value == '2': 
            dist = dst_dir + "20-39/"
        elif value == '3': 
            dist = dst_dir + "40-69/"
        else:
            dist = dst_dir + "70+/"
        dist += key + '.jpg'
        shutil.copy(img_dir, dist)

if __name__ == '__main__':
    split_data()