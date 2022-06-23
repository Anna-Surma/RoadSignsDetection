import numpy as np
import cv2
import os
from sklearn.ensemble import RandomForestClassifier
import bs4 as bs
import pandas as pd
import csv

path = os.getcwd()
upperPath = os.path.abspath(os.path.join(path, os.pardir))
train_annotations_path = upperPath + "/train/annotations/"
train_images_path = upperPath + "/train/images/"
train_annotations_files = os.listdir(train_annotations_path)
train_images_files = os.listdir(train_images_path)
test_images_path = upperPath + "/test/images/"
test_images_files = os.listdir(test_images_path)

def create_train_csv():
    data_xml = []
    for image in train_annotations_files:
        annot_xml_path = train_annotations_path + image
        soup = bs.BeautifulSoup(open(annot_xml_path, "r").read(), 'xml')
        image_size = soup.find('size')
        image_shape = [image_size.find('width').text, image_size.find('height').text]
        for object in soup.find_all('object'):
            bndbox = object.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)

            x_size = int(xmax - xmin)
            y_size = int(ymax - ymin)

            if ((x_size < int(image_shape[0]) / 10) or (y_size < int(image_shape[1]) / 10)):
                continue
            else:
                name = object.find('name').text
                filename = soup.find('filename').text
                if(name == 'speedlimit'):
                    type = 1
                else:
                    type = 2
                if (name != 'speedlimit'):
                    name = 'other'
                xml_val = [filename, name, type, xmin,  xmax, ymin, ymax]
                # TODO Nie łatwiej to wrzucić do jakiejś struktury, np. słowinka, zamiast zapisywać na dysku w innym formacie?
                data_xml.append(xml_val)
                col_names = ['filename', 'name', 'type', 'x_min', 'x_max', 'y_min',  'y_max']
                df = pd.DataFrame(data_xml, columns=col_names)
                df.to_csv(path + '/X_train.csv', index=False)


def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

def gen_sift_features(gray_img):
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def BOW():
    # TODO Słownik można zapisać na dysku zamiast go tworzyć od nowa za każdym razem.
    delete_first = 0
    dictionarySize = 100
    bow = cv2.BOWKMeansTrainer(dictionarySize)

    with open(path + '/X_train.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            if (delete_first == 0):
                delete_first = 1
            else:
                image = cv2.imread(train_images_path + row[0])
                image_part = image[int(row[5]): int(row[6]), int(row[3]): int(row[4])]
                kp, dsc = gen_sift_features(to_gray(image_part))

                if dsc is not None:
                    bow.add(dsc)
    dictionary = bow.cluster()
    np.save('dictionary.npy', dictionary)

def prepere_bow(sift):
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    dictionary = np.load('dictionary.npy')
    bow.setVocabulary(dictionary)
    return bow

def extract_train():
    sift = cv2.SIFT_create()
    delete_first = 0
    bow = prepere_bow(sift)
    train_desc_tab = []

    with open(path + '/X_train.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            if (delete_first == 0):
                delete_first = 1
            else:
                image = cv2.imread(train_images_path + row[0])
                sightPart = image[int(row[5]): int(row[6]), int(row[3]): int(row[4])]
                descriptor = bow.compute(to_gray(sightPart), sift.detect(to_gray(sightPart)))

                if descriptor is not None:
                    partDicionary = {'descryptors': descriptor}
                    train_desc_tab.append(partDicionary)
                else:
                    # TODO Lepiej w ogole pominac takie przypadki.
                    partDicionary = {'descryptors': np.zeros((1, 100))}
                    train_desc_tab.append(partDicionary)
    return train_desc_tab

def train(train_desc_tab):
    types = []
    train_desc = np.empty((1, 100))
    delete_first = 0

    with open(path + '/X_train.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            if (delete_first == 0):
                delete_first = 1
            else:
                types.append(row[2])

        for i in range(len(train_desc_tab)):
            train_desc = np.vstack((train_desc, train_desc_tab[i]["descryptors"]))

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_desc[1:], types)
    return clf

def create_input_csv():
    data_input = []
    file_nr = int(input())
    for numer in range(file_nr):
        filename = str(input())
        bndbox_nr = int(input())
        for j in range(bndbox_nr):
            xmin, xmax, ymin, ymax = input().split()
            l = [filename, xmin, xmax, ymin, ymax]
            data_input.append(l)

    col_names = ['filename', 'x_min', 'x_max', 'y_min', 'y_max']
    df = pd.DataFrame(data_input, columns=col_names)
    df.to_csv(path + '/X_test.csv', index=False)

def extract_input():
    sift = cv2.SIFT_create()
    delete_first = 0
    bow = prepere_bow(sift)
    desc_input_tab = []

    with open(path + '/X_test.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            if (delete_first == 0):
                delete_first = 1
            else:
                image = cv2.imread(test_images_path + row[0])
                sightPart = image[int(row[3]): int(row[4]), int(row[1]): int(row[2])]
                descriptor = bow.compute(to_gray(sightPart), sift.detect(to_gray(sightPart)))

                if descriptor is not None:
                    partDicionary = {'descryptors': descriptor}
                    desc_input_tab.append(partDicionary)
                else:
                    partDicionary = {'descryptors': np.zeros((1, 100))}
                    desc_input_tab.append(partDicionary)
    return desc_input_tab

def predict_im(train_data, desc_input_tab):
    pred_desc_tab = []
    for i in desc_input_tab:
        partDicionary = {'predictedStatus': train_data.predict(i['descryptors'])}
        pred_desc_tab.append(partDicionary)
    return pred_desc_tab

if __name__ == '__main__':
    x = input()

    create_train_csv()
    BOW()
    train_desc_tab = extract_train()
    train_data = train(train_desc_tab)

    if x == "classify":
        create_input_csv()
        desc_input_tab = extract_input()
        dataClassify = predict_im(train_data, desc_input_tab)
        for i in range(len(dataClassify)):
            if (dataClassify[i]["predictedStatus"]) == str(1):
                print("speedlimit")
            else:
                print("other")