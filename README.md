<p align="center">
  <img src="https://user-images.githubusercontent.com/66112225/195411937-9406438a-f073-4602-a41c-133d6f7c4372.png" />
</p>
<h2 align="center">ROAD SIGNS DETECTION </h2>

<p align="center">
  The project creates a system that detects speed limit signs in photos using Random forest, i.e. a machine learning algorithm.
</p>

## How to run
Data for testing and training should be placed in the directory superior to the main directory of the project, in the train and test folders, in which the annotations and images subfolders will be located. 
File structure:

![Zrzut ekranu 2022-10-12 203300](https://user-images.githubusercontent.com/66112225/195421083-17daf9cb-ccc0-405d-817e-cc4a0ea21ab1.png)

The annotations directory should contain .xml files describing the objects in the photo for use only during training. The structure of the xml file:

![Zrzut ekranu 2022-10-12 203532](https://user-images.githubusercontent.com/66112225/195421549-8ad02fae-0854-4b2f-a27a-4a97a4b60cf7.png)

After starting the program on the standard input, enter the string classify and information about the image slices to be classified. Input format:

![Zrzut ekranu 2022-10-12 203804](https://user-images.githubusercontent.com/66112225/195421948-9aee657c-595e-4297-9785-627e2826b815.png)

Gdzie: 
- $n_{files}$ - the number of files to be processed

- $file_1$ -name of the 1st photo, 

- $n_1$ - the number of image slices to be classified in the 1st image, 

- $xmin_{1_1}$ $xmax_{1_1}$ $ymin_{1_1}$ $ymax_{1_1}$ - the coordinates of the rectangle containing the first image slice in the 1st photo.

Sample input:

![Zrzut ekranu 2022-10-12 205006](https://user-images.githubusercontent.com/66112225/195424214-3aff1857-be60-4130-8667-8e6c72b6ff2d.png)

## Tech stack
- Python  
- Scikit-learn 
  - Random Forest 
- OpenCV 
- Beautiful Soup 4  
- Pandas 

## Description
After starting, the program performs training based on the data in the train folder.

The effectiveness of the classification of image slices is checked by dividing into 2 classes: speedlimit (any speed limit sign) and other (no speed limit sign).

Sample output: 

![Zrzut ekranu 2022-10-12 210103](https://user-images.githubusercontent.com/66112225/195426256-db53c50a-9bd3-4341-9840-2753589a3b5a.png)

