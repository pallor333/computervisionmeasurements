Requirements:

pip install opencv-python

pip install numpy

pip install matplotlib

pip install imutils

pip install rembg



This program takes in three parameters as a command line argument, (1) the image of the t-shirt, (2) the image of the object and (3) the name of the object. The t-shirt and the object should be from the same base image. Using the object as reference, the program returns the approximate measurements of the t-shirt. Inside the program is a dictionary of two objects (quarter, and playing card) with the values of it's width and length. The last command line argument specifies which object's measurements are to be used.



Several images are output by the program: the removeBG images can be ignored but they are essential to the program functioning. The keypoint jpgs show what key points, the points used for measurements, the program has found. It's good for visual confirmation and seeing what went wrong; the the program does not always work correctly.



Sample images are provided in the 'sample' folder. T-shirts are labelled as [color][number], e.g. "purple1.jpg". Objects are labelled[color][number][object] and correspond to the same color and numbered shirt. For example 'purple1coin.jpg' corresponds to 'purple1.jpg'. The two objects used are a U.S Quarter and a standard playing card. Actual measurements of the four different t-shirts are listed in 'shirtmeasurements.txt' in the samples folder. Not all samples are suppose to work, some are purposefully incorrect to show the shortcomings of the program.



I couldn't get object detection to work, so if the user wants to try their own images, they must manually crop the image for t-shirt and object. Remember that the t-shirt and object images need to be from the same image when loaded into the program.



When taking a picture the user should make sure:

- Image should only contain t-shirt and object of reference (quarter or card).

- Image can contain more than one OOR if you crop them all separately, I have done this for several images in the samples folder.

- T-shirt should be laid as flat as possible, especially around the edges.

- The entirety of the t-shirt and object must be in frame and there must be some buffer room around the edges of the shirt or object. In other words, the edge of the t-shirt or object should not be right up against the frame of the photo.

- Photo must be vertical, the t-shirt top must be pointed towards the top of the photo, the bottom of the t-shirt must be pointed to the bottom of the photo, etc. The same goes for the object.

- Object of reference must not directly overlap with the t-shirt, there must be room between the two.

- Background should be monochromatic, pattern-less and contrasting with the color of the shirt. Ideally, the background should contrast as much as possible with the color of the shirt. This is because the edge detecting algorithm struggles to detect edges when the shirt and background are too similar in color.

- Photo should be well lit with no shadows on or around the shirt.
# Measurements-with-Computer-Vision-
# computervisionmeasurements
# computervisionmeasurements
