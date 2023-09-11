import cv2 
import numpy as np
import os
from matplotlib import pyplot as pt
   
#Canny edge detection function to get edges 
def canny(img):
    #Extract the blue plane from input image and perform canny edge detection on the extracted blue plane
    blue = img[:,:,0]
    edges = cv2.Canny(blue, 16 , 186)
    #Apply morphological closing on edges using a 17x17 structuring element
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, se)
    return closing

#Floof fill to fill the gap to get the mask
def fillhole(img):
    floodfill = img.copy()
    #Apply two pixels more than the input image and create a mask for flood fill
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    floodfill = np.uint8(floodfill)
    #Perform flood fill from point 100, 100 with white color and invert the filled image
    cv2.floodFill(floodfill, mask, (100, 100), 255)
    invert = cv2.bitwise_not(floodfill)
    #Combine original and inverted image to fill the hole and invert again
    img_out = img | invert
    img_out = cv2.bitwise_not(img_out)
    #Apply dilation to refine the mask with kernel 7x7 kernel size
    kernel = np.ones((7,7), np.uint8)
    dilate = cv2.dilate(img_out, kernel, iterations = 1)
    #Perform morphological closing to remove small region usign 7x7 structuring element
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    predicted_mask = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, se)
    return predicted_mask 

#Drawing contours for skyline
def skylineDetection(mask):
    #Detect skyline in the predicted mask using canny edge detection
    edges = cv2.Canny(mask, 50, 150)
    contours, hierarchy= cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    skyline_contours = []
    for contour in contours:
        #Calculate the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        #Filter out cloud contours that are unlikely to be the skyline
        if h > 10 and w > 200:
            skyline_contours.append(contour)
    
    skyline = np.zeros_like(edges)
    cv2.drawContours(skyline, skyline_contours, -1, 255, 3)
    return skyline
   
#Day and night image classifier    
def dayOrNight(img):
    #Convert the image to grayscale then perform otsu thresholding to create a binary image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bi_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #Calculate the number of white pixels in the binary image
    no_white_px = cv2.countNonZero(bi_img)
    h, w = bi_img.shape
    white_px = (no_white_px / (h * w)) * 100
    threshold = 20
    #Determine whether is day or night based on the number of white pixels
    if white_px > threshold:
        return 'Day Time'
    else:
        return 'Night Time' 

#Accuracy calculation using MSE
def calAccuracy(out, expected):
    #Calculate the MSE between the predicted and actual mask and then calculate the accuracy
    sqr_diff = (out - expected) ** 2
    mse = np.mean(sqr_diff)
    accuracy = 1/(1 + mse)
    return accuracy

#Night image enhancement
def nightEnhancement(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv_img)
    
    adj_hue = (h + 70) % 180
    adj_saturation = np.clip(s * 0.6, 10, 100)
    adj_saturation = np.uint8(adj_saturation)
    
    adj_hsv = cv2.merge([adj_hue, adj_saturation, v])
    adj_img = cv2.cvtColor(adj_hsv, cv2.COLOR_HSV2BGR)
    return adj_img

#Main function
def main(datasets, file, expected_dic):
    img = cv2.imread(f'datasets/{datasets}/{file}', 1)
    #Image to convert to rgb and read the actual mask from dictionary for evaluation then determine day or night image
    ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    actual_mask = cv2.imread(expected_dic[datasets], 0)
    actual_skyline = skylineDetection(actual_mask)
    time = dayOrNight(img)
    # If it is not a day image apply night image enhancement
    if time != 'Day Time':
        img = nightEnhancement(img)
    #Perform edge detection and flood fill to get predicted mask then perform another edge detection for skyline
    edges = canny(img)
    predicted_mask = fillhole(edges)
    predicted_skyline = skylineDetection(predicted_mask)
    
    #Display the image 
    pt.figure()
    pt.subplot(1,3,1)
    pt.title('Original: ' + time)
    pt.imshow(ori)
    pt.subplot(1,3,2)
    pt.title('Mask (Ground Truth)')
    pt.imshow(predicted_mask, cmap = 'gray')
    pt.subplot(1,3,3)
    pt.title('Skyline')
    pt.imshow(predicted_skyline, cmap='gray')
    cv2.waitKey(1000)
    cv2.destroyAllWindows()  
    
    #Save the output figure into a new folder call result and save into their respective datasets number folder
    current_path = os.getcwd()
    save_path = os.path.join(current_path, f'results/{datasets}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path = (f'results/{datasets}/{file}')
    filename = os.path.splitext(path)[0]
    pt.savefig(f'{filename}_result.jpg')
    pt.close()
    
    mask_acc = calAccuracy(predicted_mask, actual_mask)
    skyline_acc = calAccuracy(predicted_skyline, actual_skyline)
    return mask_acc, skyline_acc

#Interpreter
if __name__ == '__main__': 
    current_path = os.getcwd()
    datasets = ['623', '684','9730', '10917']
    expected_dic = {'623':'mask/623.png', 
                    '684':'mask/684.png', 
                    '9730':'mask/9730.png', 
                    '10917':'mask/10917.png'}
    avg_mlist = []
    avg_slist = []
    #For loop for reading the datasets folder and each dataset number folder then calculate their specific average 
    #accuracy and overall average accuracy for mask and skyline
    for data in datasets:
        acc_mlist = []
        acc_slist = []
        path = os.path.join(current_path, f'datasets/{data}')
        for file in os.listdir(path):
            mask_acc, skyline_acc = main(data, file, expected_dic)
            acc_mlist.append(mask_acc)
            acc_slist.append(skyline_acc)
        
        #Error handling for none type object
        acc_mlist = [i for i in acc_mlist if i is not None]
        acc_slist = [i for i in acc_slist if i is not None]
        
        avg_mask_acc = np.mean(acc_mlist)  
        avg_skyline_acc = np.mean(acc_slist)
        
        avg_mlist.append(avg_mask_acc)
        avg_slist.append(avg_skyline_acc)
        print(f'Average mask accuracy of {data} dataset: {avg_mask_acc:.2f}')   
        print(f'Average skyline accuracy of {data} dataset: {avg_skyline_acc:.2f}')   
        print()
    
    ttl_mask_acc = np.mean(avg_mlist)
    ttl_skyline_acc = np.mean(avg_slist)
    print(f'Total average mask accuracy of the system: {ttl_mask_acc:.2f}')
    print(f'Total average skyline accuracy of the system: {ttl_skyline_acc:.2f}')