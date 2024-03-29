import os
import glob

#path to file with labels
label_dir = 'train/labels/'

img_width = 608
img_height = 608

#create new file
with open('gt_train.txt', 'w') as gt_file:

    for label_file in glob.glob(os.path.join(label_dir, '*.txt')):

        with open(label_file, 'r') as f:
            lines = f.readlines()
        

        for line in lines:

            elements = line.strip().split(' ')
            
            #transform yolov5 format to gt.txt
            class_id = elements[0]
            x_center = float(elements[1])
            y_center = float(elements[2])
            width = float(elements[3])
            height = float(elements[4])
            
            x_min = int((x_center - width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            x_max = int((x_center + width / 2) * img_width)
            y_max = int((y_center + height / 2) * img_height)
            
            gt_file.write(f'{os.path.basename(label_file)[:-4]}.jpg;{x_min};{y_min};{x_max};{y_max};{class_id+1}\n')
