import numpy as np
import pandas as pd
import skimage
from skimage import data
from skimage.util.shape import view_as_blocks
from skimage import io
import os
import re
import sys

"""This is our reference table for the NIST dataset to map manipulated images to their heatmaps that display where the actual manipulations occured"""
#df = pd.read_csv('NC2016-manipulation-ref_converted.csv') 
df = pd.read_csv('/home/robert/NC2016_Test0613/reference/manipulation/NC2016-manipulation-ref.csv', delimiter='|')

"""This is our removal reference table for the NIST dataset, used for mapping manipulated images to their removal heatmaps"""
df_removal = pd.read_csv('/home/robert/NC2016_Test0613/reference/removal/NC2016-removal-ref.csv', delimiter='|')

"""This is our splice reference table for the NIST dataset, used for mapping manipulated images to their splice heatmaps"""
df_splice = pd.read_csv('/home/robert/NC2016_Test0613/reference/splice/NC2016-splice-ref.csv', delimiter='|')

df['IsManipulationTypeRemoval'] = df['IsManipulationTypeRemoval'].map({'N': 0, 'Y': 1})
df['IsManipulationTypeSplice'] = df['IsManipulationTypeSplice'].map({'N': 0, 'Y': 1})
df['IsManipulationTypeCopyClone'] = df['IsManipulationTypeCopyClone'].map({'N': 0, 'Y': 1})
df['Lighting'] = df['Lighting'].map({'same': 0, 'different': 1})
probe_file_names = df.ix[:,2].copy()
probe_removal_mask_file_names = df_removal.ix[:,3].copy()
print('Removal Mask Files')
print(probe_removal_mask_file_names)
probe_removal_mask_file_names.fillna(0,inplace=True)
print(probe_removal_mask_file_names)
probe_splice_mask_file_names = df_splice.ix[:,[1,3]].copy()
print('Splice Mask Files')
print(probe_splice_mask_file_names)
probe_splice_mask_file_names.fillna(0,inplace=True)
print(probe_splice_mask_file_names)
probe_mask_file_names = df.ix[:,3].copy()
print('CopyClone Mask Files')
print(probe_mask_file_names)
probe_mask_file_names.fillna(0, inplace=True)
print(probe_mask_file_names)
probe_manipulation_removal = df.ix[:,11].copy()
probe_manipulation_splice = df.ix[:,12].copy()
probe_manipulation_copyclone = df.ix[:,13].copy()
probe_lighting_changes = df.ix[:,16].copy()
probe_lighting_changes.fillna(0, inplace=True)


"""Create and initialize dataframe to hold data for all manipulated images"""
nist_data = pd.DataFrame
image_info = {}

"""Number of rows and columns we will break each image into"""
rows = 4
cols = 4

probe_ids = []
for image in list(probe_file_names):
    probe_id = re.sub(r'probe/', r'', image)
    probe_id = re.sub(r'.jpg', r'', probe_id)
    probe_ids.append(probe_id)

probe_removal_mask_files = []
for image in list(probe_removal_mask_file_names):
    probe_removal_mask_files.append(image)

"""Now we will go through all the reference file rows and for each probe image, if there is no splice mask file for any combination of that probe file and a world file then the image does not have a corresponding splice heatmap and gets a value of 255 (which indicates there should be a heatmap with no splice indication.)"""
print("Preparing to find Splice mask files")
probe_splice_mask_dict = {}
for image in probe_ids:
    probe_splice_mask_dict[image] = 0
"""If there are duplicate probe file ids in the reference file"""
if(len(probe_ids) != len(probe_splice_mask_dict)): 
    print("Duplicate probe id files, exiting...")
    sys.exit()
probe_splice_mask_files = []
x = 0
for row in probe_splice_mask_file_names.iterrows():
    if(str(row[1]['ProbeMaskFileName']) != str(0)):
        print(row[1]['ProbeMaskFileName'])
        probe_splice_mask_dict[row[1]['ProbeFileID']] = row[1]['ProbeMaskFileName']
        x += 1
    #    probe_splice_mask_files.append(imagdde)
print(x)
"""Now that we have a dictionary listing all the images that have a splice masks, create a list of the splice mask files. There should be x/2 non zero entries. """
y = 0
for image in probe_ids:
    probe_splice_mask_files.append(probe_splice_mask_dict[image])
    if(probe_splice_mask_dict[image] != 0):
        y += 1
print(y)

probe_mask_files = []
for image in list(probe_mask_file_names):
    probe_mask_files.append(image)

probe_removal = []
for i in list(probe_manipulation_removal):
    probe_removal.append(i)

probe_splice = []
for i in list(probe_manipulation_splice):
    probe_splice.append(i)

probe_copyclone = []
for i in list(probe_manipulation_copyclone):
    probe_copyclone.append(i)

probe_lighting = []
for i in list(probe_lighting_changes):
    probe_lighting.append(i)

if(len(probe_ids) != len(probe_mask_files)):
    print('Error, mismatch length of probe ids and mask files')
    sys.exit()

if(len(probe_ids) != len(probe_removal_mask_files)):
    print('Error, mismatch length of probe ids and removal mask files')
    sys.exit()

"""We expect the manipulation (copyCLone) and removal mask reference files to have the same number of lines. However, we expect the number of lines in the splice reference to be much greater than the number of lines in the manipulation and removal reference files because it has a row for each combination in the cross product of probe and world files. """
if(len(probe_ids) != len(probe_splice_mask_files)):
    print('Error, mismatch length of probe ids and splice mask files')
    sys.exit()
    

for i in range(len(probe_ids)):
    image_info[probe_ids[i]] = {'mask_file': 0} # I think this line is redundent. Flagged for removal later.
    image_info[probe_ids[i]] = {'mask_file': probe_mask_files[i]}
    image_info[probe_ids[i]]['removal_mask_file'] = probe_removal_mask_files[i]
    image_info[probe_ids[i]]['splice_mask_file'] = probe_splice_mask_files[i]
    image_info[probe_ids[i]]['removal'] = probe_removal[i]
    image_info[probe_ids[i]]['splice'] = probe_splice[i]
    image_info[probe_ids[i]]['copyclone'] = probe_copyclone[i]
    image_info[probe_ids[i]]['lighting'] = probe_lighting[i]
    for j in range(rows):
        for k in range(cols):
            image_info[probe_ids[i]][j*rows + k] = {}

print('Adding global algorithm scores')
t1_algorithms = ['abb01', 'block01', 'block02', 'cfa01', 'cfa02', 'combo01', 'copymove01', 'dct01', 'dct02', 'dct03_A', 'dct03_NA', 'ela01', 'noise01', 'noise02']
"""Add global algorithm score for each image."""
for algor in t1_algorithms:
    print('Finding scores for: ' + algor)
    for part in range(1,4):
        print('Checking \'exp_pt' + str(part) + '\'')
        scores_file_path = '/home/robert/exp_pt' + str(part) + '/' + algor + '/' + algor + '.csv'
        try:
            scores_df = pd.read_csv(scores_file_path, delimiter='|')
        except IOError:
            continue
        for image in probe_ids:
            image_info[image][algor] = float('nan')
            for row in scores_df.iterrows():
                if(row[1]['ProbeFileID'] == image):
                    image_info[image][algor] = row[1]['ConfidenceScore']
                    continue

        
print('Initializing heatmap region values')
"""Initialize each image to have heat map values of 255 (no detection) for every segment), this is imperfect and will be replaced at a later date."""
for image in probe_ids:
    for algor in t1_algorithms:
        for i in range(rows):
            for j in range(cols):
                image_info[image][i*rows + j][algor] = 255
    for i in range(rows):
        for j in range(cols):
            image_info[image][i*rows + j]['manipulation'] = 255
            image_info[image][i*rows + j]['removal'] = 255
            image_info[image][i*rows + j]['splice'] = 255
        
print('Storing regional data from heatmaps')
for image in probe_ids:
    for algor in t1_algorithms:
        print('Checking ' + image + ' for ' + algor + ' heat map...')
        flag = False
        for part in range(1,4):
            path = '/home/robert/exp_pt' + str(part) + '/' + algor + '/mask'
            if(os.path.isdir(path)):
                directory_files = os.listdir(path)
                for dir_file in directory_files:
                    if(image in dir_file):
                        print('Heat map ' + dir_file + ' found, generating segment averages')
                        if(flag != False):
                            print('Error: More than one heatmap for same algorithm for an image')
                            #sys.exit()
                        flag = True
                        """Create average blocks for file"""
                        heat_map_file = path + '/' + dir_file
                        heat_map = io.imread(heat_map_file)
                        rpb = heat_map.shape[0]/rows # Rows per block
                        cpb = heat_map.shape[1]/cols # Columns per block
                        blocks = view_as_blocks(heat_map, block_shape=(rpb,cpb))
                        grid_pixel_averages = []
                        for i in range(rows):
                            for j in range(cols):
                                image_info[image][i*rows + j][algor] = np.amin(blocks[i][j])

print('Collecting scores')
for image in probe_ids:
#    if(image_info[image]['removal_mask_file'] != 0):
#        mask_file = image_info[image]['removal_mask_file']
#        path = '/home/robert/daniel_lowd_research/NC2016_inference/NC2016_Test0613/reference/removal/mask'
#        directory_files = os.listdir(path)
#        for dir_file in directory_files:
#            if(mask_file in dir_file):
#                mask = io.imread(mask_file)
#                rpb = mask.shape[0]/rows
#                cpb = mask.shape[1]/cols
#                blocks = view_as_blocks(image, block_shape=(rpb,cpb))
#                for i in range(rows):
#                    for j in range(cols):
#                        image_info[image][i*rows + j]['removal'] = np.average(blocks[i][j])
#
#    if(image_info[image]['splice_mask_file'] != 0):
#        mask_file = image_info[image]['splice_mask_file']
#        path = '/home/robert/daniel_lowd_research/NC2016_inference/NC2016_Test0613/reference/splice/mask'
#        directory_files = os.listdir(path)
#        for dir_file in directory_files:
#            if(mask_file in dir_file):
#                mask = io.imread(mask_file)
#                rpb = mask.shape[0]/rows
#                cpb = mask.shape[1]/cols
#                blocks = view_as_blocks(image, block_shape=(rpb,cpb))
#                for i in range(rows):
#                    for j in range(cols):
#                        image_info[image][i*rows + j]['splice'] = np.average(blocks[i][j])

    """Turns out we don't need the other two sets of heat maps because the manipulation heat maps cover all three types of manipulation, we just need to connect the actual manipulation to the heat map and give the correct manipulation that value."""
    if(image_info[image]['mask_file'] != 0):
        heatmap_manipulation_tag = ''
        if(image_info[image]['copyclone'] != 0):
            heatmap_manipulation_tag = 'copyclone'
        if(image_info[image]['removal'] != 0):
            heatmap_manipulation_tag = 'removal'
        if(image_info[image]['splice'] != 0):
            heatmap_manipulation_tag = 'splice'
        #else:
        #    print('Image has a manipulation but no heat map associated with that manipulation, this must be wrong. Exiting...')
        #    sys.exit()
        mask_file = image_info[image]['mask_file']
        path = '/home/robert/NC2016_Test0613/reference/manipulation/mask'
        directory_files = os.listdir(path)
        flag = False
        for dir_file in directory_files:
            dir_file_path = path + '/' + dir_file
            if(mask_file in dir_file_path):
                flag = True
                mask = io.imread(dir_file_path)
                rpb = mask.shape[0]/rows
                cpb = mask.shape[1]/cols
                blocks = view_as_blocks(mask, block_shape=(rpb,cpb))
                for i in range(rows):
                    for j in range(cols):
                        for manipulation_type in ['removal', 'splice', 'copyclone']:
                            if(manipulation_type == heatmap_manipulation_tag):
                                image_info[image][i*rows + j][manipulation_type] = np.amin(blocks[i][j]) # This is the average pixel value for the area where the manipulation happened.
                            else:
                                image_info[image][i*rows + j][manipulation_type] = 255 # This manipulation is not present, thus if there was a heatmap it would be all white.
                        #image_info[image][i*rows + j]['manipulation'] = np.average(blocks[i][j])
                        #image_info[image]['mask_blocks'][i*rows + j] = np.average(blocks[i][j])
        if(not flag):
            print('Mask file for image: ' + image + ' not found, looking for ' + mask_file + ', exiting...')
            sys.exit()
    else:
        for i in range(rows):
            for j in range(cols):
                for manipulation_type in ['removal', 'splice', 'copyclone']:
                    image_info[image][i*rows + j][manipulation_type] = 255


manipulation_types = ['removal', 'splice', 'copyclone']
map_man_type = {'removal': 'Removal', 'splice': 'Splice', 'copyclone': 'CopyClone'}
csv_file = 'NIST_training_data_v17.csv'
csv = open(csv_file, 'w')
schema_line = ''
schema_line += 'image,'
schema_line += 'manipulation_mask,'
schema_line += 'removal_global,'
schema_line += 'splice_global,'
schema_line += 'copyclone_global,'
schema_line += 'lighting,'
for algor in t1_algorithms:
    schema_line += algor + '_global,'
for i in range(rows):
    for j in range(cols):
        for manip_type in manipulation_types:
            schema_line += manip_type + '_hm_sec' + str(i*rows + j) + ','
        #schema_line += 'Manipulation_HM_sec' + str(i*rows + j) + ','
        for algor in t1_algorithms:
            schema_line += algor + '_sec' + str(i*rows + j) + ','
schema_line = re.sub(r',$', r'\n', schema_line)
csv.write(schema_line)
for image in probe_ids:
    line = ''
    line += image + ','
    line += str(image_info[image]['mask_file']) + ','
    line += str(image_info[image]['removal']) + ','
    line += str(image_info[image]['splice']) + ','
    line += str(image_info[image]['copyclone']) + ','
    line += str(image_info[image]['lighting']) + ','
    for algor in t1_algorithms:
        line += str(image_info[image][algor]) + ','
    for i in range(rows):
        for j in range(cols):
            for manip_type in manipulation_types:
                line += str(image_info[image][i*rows + j][manip_type]) + ',' 
            #line += str(image_info[image][i*rows + j]['manipulation']) + ','
            for algor in t1_algorithms:
                line += str(image_info[image][i*rows + j][algor]) + ','
    line = re.sub(r',$',r'\n',line)
    csv.write(line)
csv.close()
