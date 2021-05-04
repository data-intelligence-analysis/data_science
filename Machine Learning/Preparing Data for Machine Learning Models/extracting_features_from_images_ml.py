import cv2
import numpy as np
import matplotlib.pyplot as plt


###Final Data Preprocessing
'''Preparing an unorganized data set to feed the machine learning models in an unbiased way so that the model can give good results, for example we don't want the machine learning model to know that 
1st 50 records always go to the object, and the next 80 records will always go to the background and so on. So we have to mix it up for the model to learn'''
## Create a shuffling function
def shuffle(features,labels):
    arr=np.random.permutation(features.shape[0])
    features_shuffled = features[arr, :]
    shuffled_labels=labels[arr]
    ###Shuffling every record of the feature
    print("Features Shuffled Array Size:",features_shuffled.shape)
    #Shuffle the corresponding lable size
    print("Labels Shuffled Array Size:", shuffled_labels.shape)
    print("Shuffled Array labels Content", shuffled_labels)
    
    return features_shuffled, shuffled_labels

###Extracting Dataset - Extracting Features of an Image
#Function to call the extracted features
#ROI - Region of Interest
def ROI(img):

    #### Background
    '''-------------------------'''
    ###selecting a region of interest on an image
    '''cv2.selectROI('ROI',img,False) - #ROI - Window Name, img = directory to the image,
    False = The direction of the selector of the image is from upper left to lower right in a rectangular formation'''
    
    '''#convert the coordinates to a Numpy array, x = x coordinate of the rectangle we draw, y= coordinate of the rectangle we draw,
    w = width of the rectangle, h = height or length of the rectangle you draw'''
    #Slecting the Region of Interestt of the background of photo and putting it into a numpy array (essentially a list)
    x,y,w,h = np.array(cv2.selectROI('ROI',img,False)

    #use image as background and convert to a numpy array
    #3d image - 3D Size because image consists of three channels, So imagine this figure exist three times every time for each channel
    background = img[y : y+h , x : x+w] #Pixel Value of background image - Image containing all pixeks increase the height and width +h +w
    print('Background Array Size:', background.shape)
                       
    # Convert from 3D array to 2D array
    #background - array we would want to reshape
    '''background.shape[0]*background.shape[1] - the desired size we want our array to have,
    which is background.shape[0] - represents the number of rows in our original array, multiplied by background.shape[1],
    which represents the array of columns in our original array and
    then comma 3 - this represents the new number of rows in our new array and this represents the new number of columns'''
    background = np.reshape(background, (background.shape[0]*background.shape[1],3)) #Total number of pictures in the background ROI 
    print('Background New Array Size:', background.shape)


    #### Object
    '''--------------------------'''
    #Slecting the Region of interest of the object of photo and putting it into a numpy array (essentially a list)
    #convert the coordinates to a Numpy array
    x,y,w,h = np.array(cv2.selectROI('ROI',img,False)
    #use image as object and convert to a numpy array
    #3d image - 3D Size because image consists of three channels, So imagine this figure exist three times every time for each channel
    object2 = img[y : y+h, x : x+w] #Pixel Value of background image - Image containing all pixeks increase the height and width +h +w
    print("Object Array Size:", object2.shape)
    # Convert from 3D array to 2D array
    # object2 - array we would want to reshape
    object2 = np.reshape(object2, (object2.shape[0]*object2.shape[1],3))#Total number of pictures in the object ROI
    print('Object New Array Size:', object2.shape)
    

                
    #Concatenate arrays and label the dataset
    #feature=  np.concatenate() We will specify the 2 arrays. We want to add together, those are (object, background) and then axis=0' for the rows

    features = np.concatenate((object2,background),axis=0)

    print("Features Array Size:",features.shape)

    #np.full() - This function creates a numpy array and we ill fill it all with zero
    '''#object2.shpae[0] and background.shape[0] - Desired size of our array which will be equal to the number of records we have in the object array,
    The number of records we have in the object array is number of rows of the object array.'''
    object_labels=np.full(object2.shape[0],0)
    background_labels=np.full(background.shape[0],0)
    labels=np.concatenate((object_labels,background_labels),axis=0)
    #Desired size of our array which will be equal to the number of records we have in the object array, The number of records we have in the object array is number of rows of the object array.
    print("Labels Array Size:", labels.shape)
    # [0,0...1,1]
    #Background label will be number 1 and the object label will be number 0
    print("Labels Array Content:", labels)

    features,labels = shuffle(features, labels)
    
    return features, labels
                       

#Featues as Numpy arrays
    

if __name__ == '__main__':
     # 1 - Indicates you want to read images in color
     # 0 - Indicates you want to read images in black or white
     # use option "r" to fix directories with spaces inbetween them
     # cv2.imread(r'directory to image', 1)
     img = cv2.imread(r'<directory_to_image>', 1)
     #calling the function, image is returned
     features, labels=ROI(img)
     


     #you can draw a rectangle on the image,
     #which produces 4 coordinates of the rectangle we drew
     
     '''# cv2.imshow('window_name', image read from directory)
     cv2.imshow('Picture', img)

     #write zero to terminate the window manually
     cv2.waitKey(0)
     #destroy all windows
     cv2.destroyAllWindows()'''



