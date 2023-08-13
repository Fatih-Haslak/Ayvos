import os, shutil, random

# preparing the folder structure
# kod yolo/yolov5/datasets içerisinde bulunmalı
# dataset yolo/yolov5/datasets/full_data_path_name.. olarak olmalı

full_data_path = 'custom_data/' # resim ve etiketlerının beraber bulundugu folder
extension_allowed = '.jpg'    # resimlerin uzantısı
split_percentage = 70         #yüzde x' kadar train yap demek
#Aşağı kod satırı folder tree' ayarlamak ıcın duzenlenmıstır yaml dosyasının duzenıde buna göredir.
#Bu kodla calısan yaml. folderın düzeni
"""

path: ../datasets/data  # dataset root dir
train: images/training/  # train images (relative to 'path') 1281167 images
val: images/validation/  # val images (relative to 'path') 50000 images
test:  # test images (optional)
"""

images_path = 'train/images/'
if os.path.exists(images_path):
    shutil.rmtree(images_path)
os.mkdir(images_path)
    
labels_path = 'train/labels/'
if os.path.exists(labels_path):
    shutil.rmtree(labels_path)
os.mkdir(labels_path)
    
training_images_path = images_path + 'training/'
validation_images_path = images_path + 'validation/'
training_labels_path = labels_path + 'training/'
validation_labels_path = labels_path +'validation/'
    
os.mkdir(training_images_path)
os.mkdir(validation_images_path)
os.mkdir(training_labels_path)
os.mkdir(validation_labels_path)

files = []

ext_len = len(extension_allowed)

for r, d, f in os.walk(full_data_path):
    for file in f:
        if file.endswith(extension_allowed):
            strip = file[0:len(file) - ext_len]      
            files.append(strip)

random.shuffle(files)

size = len(files)                   

split = int(split_percentage * size / 100)

print("copying training data")
for i in range(split):
    strip = files[i]
                         
    image_file = strip + extension_allowed
    src_image = full_data_path + image_file
    shutil.copy(src_image, training_images_path) 
                         
    annotation_file = strip + '.txt'
    src_label = full_data_path + annotation_file
    shutil.copy(src_label, training_labels_path) 

print("copying validation data")
for i in range(split, size):
    strip = files[i]
                         
    image_file = strip + extension_allowed
    src_image = full_data_path + image_file
    shutil.copy(src_image, validation_images_path) 
                         
    annotation_file = strip + '.txt'
    src_label = full_data_path + annotation_file
    shutil.copy(src_label, validation_labels_path) 

print("finished")