# import the opencv library
import cv2
import numpy as np 

import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')
# define a video capture object
video = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    check, frame = video.read()
    
    # Modify the input data by:

    # 1. Resizing the image

    img = cv2.resize(frame,(224,224))
    
    # 2. Converting the image into Numpy array and increase dimension

    test_image = np.array(img, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis=0)
    
     # 3. Normalizing the image
    normalised_image = test_image/255.0

    
      # Predict Result
    prediction = model.predict(normalised_image)

    print("Prediction : ", prediction)
    
   # # prediction = model.predict(frame)
    
   # # print("Prediction: ",prediction)
  
    # Display the resulting frame
    cv2.imshow('Result', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        print("Closing")
        break
  
# After the loop release the cap object
video.release()

# Destroy all the windows
cv2.destroyAllWindows()