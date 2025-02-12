# Fall 2022 CS50 Final Project
from math import isclose, sqrt
from PIL import Image
from rembg import remove
from skimage.morphology import skeletonize
import cv2
import imutils
import numpy as np
import sys

# Dictionary of object of reference (OOR) real world measurements
OBJECT_OF_REF = {"quarter": {"width": 0.955, "length": 0.955},
                 "card": {"width": 2.5, "length": 3.5}}


def main():
    # Take in command line arguments for filenames
    if len(sys.argv) != 4:
        print("Usage: python tshirt.py [tshirt img] [object img] [object of reference]")
        print("E.G. python tshirt.py samples/purple1.jpg samples/purple1card.jpg card")
        sys.exit(0)
    # Store command line inputs as values
    filename = sys.argv[1]
    obj_filename = sys.argv[2]
    selectedObj = sys.argv[3]

    # Throw error if selected object is not in OOR dictionary
    if selectedObj.lower() not in OBJECT_OF_REF:
        print("Please select from one of these objects of reference:", list(OBJECT_OF_REF))
        sys.exit(0)
    objWidthInch = OBJECT_OF_REF[selectedObj]["width"]
    objLengthInch = OBJECT_OF_REF[selectedObj]["length"]

    # Read image from disk
    img = cv2.imread(filename)
    if img is None:
        print("ERROR: Could not read image")
        sys.exit(0)

    # Remove background helps prepare to find contours and edges - apply to both tshirt and OOR
    tshirt = cv2.imread(removeBG(filename, "RemoveBG_shirt.jpg"))
    objNoBg = cv2.imread(removeBG(obj_filename, "RemoveBG_object.jpg"))

    # Find pixel length/width of OOR
    pixObjLength, pixObjWidth = measure_rect_object(objNoBg)
    # Find pixel measurements of tshirt
    pixShirtChest, pixShirtLength, pixShirtShoulder = measure_shirt(tshirt)

    # Averaging the pixel length and width for hopefully a more accurate metric
    pixelPerInch = round((objLengthInch/pixObjLength + objWidthInch/pixObjWidth)/2, 2)

    # Calculating tshirt measurements and printing out
    shoulder = round(pixShirtShoulder * pixelPerInch, 2)
    chest = round(pixShirtChest * pixelPerInch, 2)
    length = round(pixShirtLength * pixelPerInch, 2)
    print("The t-shirts measurements are estimated to be")
    print(f"Shoulders: {shoulder} inches across")
    print(f"Chest: {chest} inches across")
    print(f"Length: {length} inches top to bottom")


def draw_point(img_copy, point, text):
    """Draw and label point on image"""
    font = cv2.FONT_HERSHEY_COMPLEX
    # Print text in red
    cv2.putText(img_copy, str(text), (point),
                font, 0.5, (0, 0, 255))
    return None


def measure_rect_object(img):
    """Find measurement by drawing rectangle bounding box over object"""
    # Preprocess and find largest contour
    processed = preprocess(img)
    contours = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Code adapted from https://stackoverflow.com/questions/60062044/finding-the-corners-of-a-rectangle
    # Finding largest contour
    if len(contours) == 2:
        contours = contours[0]
    else:
        contours = contours[1]
    for c in contours:
        area_thresh = 0
        area = cv2.contourArea(c)
        if area > area_thresh:
            area = area_thresh
            big_contour = c

    # Find rotated rectangle
    rectangle = cv2.minAreaRect(big_contour)
    box = np.int0(cv2.boxPoints(rectangle))

    # Finding rectangle vertices
    # Copy box to sort by ascending X values without messing with original values
    box2 = np.copy(box)
    # Similar X values determine left / right; low = left, high = right
    # Similar Y values determine top / bottom; low = top, high = bottom
    box2 = box2[box2[:, 0].argsort()]
    if np.greater(box2[0][1], box2[1][1]):
        topleft = box2[1]
        botleft = box2[0]
    else:
        botleft = box2[1]
        topleft = box2[0]
    if np.greater(box2[2][1], box2[3][1]):
        topright = box2[3]
        botright = box2[2]
    else:
        botright = box2[3]
        topright = box2[2]

    # Draw rotated rectangle and corresponding vertices on copy of img
    rotatedBox = img.copy()
    cv2.drawContours(rotatedBox, [box], 0, (0, 255, 0), 2)
    draw_point(rotatedBox, topleft, "topleft")
    draw_point(rotatedBox, topright, "topright")
    draw_point(rotatedBox, botleft, "botleft")
    draw_point(rotatedBox, botright, "botright")
    cv2.imwrite('keypoints_object.jpg', rotatedBox)

    # Distance formula: sqrt( [x2-x1]^2 + [y2-y1]^2 )
    pixelwidth = sqrt((topleft[0]-topright[0])**2 + (topleft[1]-topright[1])**2)
    pixellength = sqrt((topleft[0]-botleft[0])**2 + (topleft[1]-botleft[1])**2)

    return round(pixelwidth, 2), round(pixellength, 2)


def measure_shirt(tshirt):
    """Use Harris corner detection to find key points of tshirt """
    # OpenCV tutorial: https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
    # Corner Harris function to find corners (areas of interest)
    cornered = tshirt
    gray = np.float32(preprocess(cornered))
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # Result is dilated for marking the corners, not required
    dst = cv2.dilate(dst, None)
    # cv2.cornerHarris return value is confidence that a point==corner
    # Confidence threshold varies depending on the image. Any corner detected will be
    # marked as a corner if the confidence > 1% max probability
    ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)

    # https://stackoverflow.com/questions/44101894/extraction-of-coordinates-of-corners-using-harris-corner-detection-and-also-ret
    # Finding coordinates of corners found by Harris Corner Detection
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    # Add corner coordinate points to image
    cornered[dst > 0.01*dst.max()] = [0, 255, 0]

    # Numpy functions find indices of min/max of Y coordinate
    # Arr[:] selects the entire array, arr[,1] selects Y value
    top = corners[corners[:, 1].argmin()]
    bot = corners[corners[:, 1].argmax()]
    # Find min/max of X coordinate
    left = corners[corners[:, 0].argmin()]
    right = corners[corners[:, 0].argmax()]
    # LBC/RBC[Y] = bot[Y]; LBC = max[X] to make comparisons easier
    leftBotCorner = [np.amax(corners[:, 0]), bot[1]]
    rightBotCorner = [0, bot[1]]

    # Get indexes of coordinates for slicing into array later
    leftIndex, lbcIndex, rightIndex, rbcIndex = 0, 0, 0, 0

    # Find left bottom corner and right bottom corner of shirt
    for idx, pixel in enumerate(corners):
        # rbc[Y] and lbc[Y] ~= bot[Y]
        if isclose(pixel[1], bot[1], rel_tol=0.05):
            if pixel[0] < leftBotCorner[0]:
                leftBotCorner = pixel
                lbcIndex = idx
            if pixel[0] > rightBotCorner[0]:
                rightBotCorner = pixel
                rbcIndex = idx
        if pixel[0] == left[0] and pixel[1] == left[1]:
            leftIndex = idx
        if pixel[0] == right[0] and pixel[1] == right[1]:
            rightIndex = idx

    # Using bottom corner coordinates, find shoulder coordinates
    leftShoulder = [0, leftBotCorner[1]]
    rightShoulder = rightBotCorner
    # Left shoulder btw 0th element and left so slice into array
    for pixel in corners[0:leftIndex]:
        # left/rightShoulder[X] ~= lbc/rbc[X] but Pit[Y] is not near lbc/rbc[Y]
        if isclose(pixel[0], leftBotCorner[0], rel_tol=0.10) and not isclose(pixel[1], leftBotCorner[1], rel_tol=0.5):
            # Larger X is closer to shirt from the left
            if pixel[0] > leftShoulder[0]:
                leftShoulder = pixel
    # Right shoulder btw 0th element and right so slice into array
    for pixel in corners[0:rightIndex]:
        if isclose(pixel[0], rightBotCorner[0], rel_tol=0.05) and not isclose(pixel[1], rightBotCorner[1], rel_tol=0.5):
            # Smaller X is closer to shirt from the right
            if pixel[0] < rightShoulder[0]:
                rightShoulder = pixel

    # Using bottom corner coordinates, find armpit coordinates
    leftPit = [0, leftBotCorner[1]]
    rightPit = rightBotCorner
    # Left pit is bound between left and lbc so slice into array
    for pixel in corners[leftIndex:lbcIndex]:
        # LeftPit[X] ~= lbc[X] but Pit[Y] is not near lbc[Y]
        if isclose(pixel[0], leftBotCorner[0], rel_tol=0.10) and not isclose(pixel[1], leftBotCorner[1], rel_tol=0.05):
            # Larger X is closer to shirt from the left
            if pixel[0] > leftPit[0]:
                leftPit = pixel
    # Right pit is bound between right and rbc so slice into array
    for pixel in corners[rightIndex:rbcIndex]:
        # rightPit[X] ~= rbc[X] but Pit[Y] is not near rbc[Y]
        if isclose(pixel[0], rightBotCorner[0], rel_tol=0.10) and not isclose(pixel[1], rightBotCorner[1], rel_tol=0.05):
            # Smaller X is closer to shirt from the right
            if pixel[0] < rightPit[0]:
                rightPit = pixel

    # Use contour/corner points to calculate shirt measurements
    length = bot[1] - top[1]
    chest = rightPit[0] - leftPit[0]
    shoulder = rightShoulder[0] - leftShoulder[0]

    # Draw points on shirt for visual inspection
    draw_point(cornered, (int(top[0]), int(top[1])), "top")
    draw_point(cornered, (int(leftBotCorner[0]), int(leftBotCorner[1])), "lbc")
    draw_point(cornered, (int(rightBotCorner[0]), int(rightBotCorner[1])), "rbc")
    draw_point(cornered, (int(left[0]), int(left[1])), "left")
    draw_point(cornered, (int(right[0]), int(right[1])), "right")
    draw_point(cornered, (int(leftPit[0]), int(leftPit[1])), "leftPit")
    draw_point(cornered, (int(rightPit[0]), int(rightPit[1])), "rightPit")
    draw_point(cornered, (int(leftShoulder[0]), int(leftShoulder[1])), "leftShoulder")
    draw_point(cornered, (int(rightShoulder[0]), int(rightShoulder[1])), "rightShoulder")
    cv2.imwrite('keypoints_tshirt.jpg', cornered)

    return chest, length, shoulder


def preprocess(img):
    """Manipulate image to prepare to find contours of shirt"""
    # Convert to grayscale & remove noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Bilateral filtering esp important - smooths image without removing edges
    gray = cv2.bilateralFilter(gray, 15, 80, 80)

    # Canny returns b/w image of edges in white, rest of img in black
    # Dilate blows up edges to close gaps between them,
    # Erode shrink back down to their normal size
    edges = imutils.auto_canny(gray)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)

    return edges


def removeBG(filename, output_path):
    """Remove background then paste image onto white background to remove transparency"""
    # Remove background
    input_path = filename
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)
    # Remove transparency
    image = Image.open(output_path)
    # Create a blank background image
    bg = Image.new('RGB', image.size, (255, 255, 255))
    # Paste image to background image
    bg.paste(image, (0, 0), image)
    # Save pasted image
    bg.save(output_path)

    # Return filename of no bg image
    return output_path


main()