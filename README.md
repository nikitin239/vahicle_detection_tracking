

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_test_images/vehicle_hog.png
[image2]: ./output_test_images/Figure_1.png
[image3]: ./output_test_images/Figure_2.png
[image4]: ./output_test_images/Figure_3.png
[image5]: ./output_test_images/Figure_4.png
[image6]: ./output_test_images/Figure_5.png
[image7]: ./output_test_images/Figure_6.png
[video1]: ./output_test_images/video_out_final.mp4


### DataSet
At first I've prepared dataset for vehicles and non-vehicles features. 
I used Kitti+GTI pictures for vehicles and Extras+GTI pictures for non-vehicles.
```python
6800 vehicles
6800 non-vehicles
Shape= (64, 64, 3)
Type= float32

```
This data was downloaded using function:
```python
def getDataset():
    non_cars_files = glob.glob('/home/dnikitin/GIT/vehicledetectionandtracking/non-vehicles/non-vehicles/Extras/*.png')
    cars_files=glob.glob('/home/dnikitin/GIT/vehicledetectionandtracking/vehicles/vehicles/KITTI_extracted/*.png')
    cars = []
    not_cars = []
    for car in cars_files:
        cars.append(car)
    for non_car in non_cars_files:
        not_cars.append(non_car)
    ## Uncomment if you need to reduce the sample size
    sample_size = 6800
    cars = cars[0:sample_size]
    not_cars = not_cars[0:sample_size]
    print(len(cars))
    print(len(not_cars))
    ex = mpimg.imread(cars[0])
    print('Shape=',ex.shape)
    print('Type=',ex.dtype)

    # Return data_dict
    return cars, not_cars
```
---

### Histogram of Oriented Gradients (HOG)+Features

I've used color histogram features with color features and HOG.
all this data was processed with:
```python
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True: # Call with two outputs if vis==True to visualize the HOG
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      # Otherwise call with one output
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Spatial features calculation
def bin_spatial(img, size=(32, 32)):
    return cv2.resize(img, size).ravel()


#Calculating histogram features
def color_hist(img, nbins=32):
    ch1 = np.histogram(img[:,:,0], bins=nbins, range=(0, 256))[0]
    ch2 = np.histogram(img[:,:,1], bins=nbins, range=(0, 256))[0]
    ch3 = np.histogram(img[:,:,2], bins=nbins, range=(0, 256))[0]
    hist_features = np.concatenate((ch1, ch2, ch3))
    return hist_features
```
Spatial binning allows us to change picture resolution and still get information about picture.
At first we change picture resolution to 32x32 pix and then use ravel() function to get vector with features. 

Color histogram is one more way to extract features from picture. Color histogram shows the color level for each individual level in channel.

#### HOG- histogram of oriented gradients 
Is powerful method to extract picture features. In the HOG histograms of directions of gradients are used as features. We could calculate HOG for one channel or could calculate for each channel and then concatenate. Gradients are very useful because the magnitude of gradients is large around edges which could help us in vehicle recognition. 

![alt text][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.
I've got best results with following parameters:
```python
	color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block

    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    hog_channel = 'ALL' 
```

I've tried LUV color scheme, HLS and RGB, but best is YCrCb with calculating HOG for ALL channels.
At first I've used 8 hog orientations but 9 is better for improving accuracy

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using LinearSVC with standard parameters:loss='squared_hinge',C=1.0
after calculating all dataset features I've applied normalization:
```python
X = np.vstack((car_features, notcar_features)).astype(np.float64)

    X_scaler = StandardScaler().fit(X)  # Fit a per-column scaler
    scaled_X = X_scaler.transform(X)  # Apply the scaler to X
```
and splitted dataset to train and test part:
```python
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
```
Train results:
```python
Using spatial binning of: (32, 32) and 32 histogram bins
Using HOG: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
4.2 Seconds to train SVC...
Test Accuracy of SVC =  0.9923
My SVC predicts:  [ 0.  0.  0.  0.  1.  0.  1.  0.  1.  0.]
For these 10 labels:  [ 0.  0.  0.  0.  1.  0.  1.  0.  1.  0.]
0.00084 Seconds to predict 10 labels with SVC
```
Classifier accuracy is around 99 percent and that's great!
### Sliding Window Search

Sliding window is a method with going throw picture with fixed width and height sliding across image. I've realized tuned find cars method:
```python
def find_cars(img, ystart, ystop,xstart,xstop, scale):
    boxes = []

    img = img.astype(np.float32) / 255
    img_tosearch = img[ystart:ystop, xstart:xstop, :]
    ctrans_tosearch=cv2.cvtColor(img_tosearch,cv2.COLOR_RGB2YCR_CB)


    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    # Compute individual channel HOG features for the entire image
    hog1 = hist.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = hist.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = hist.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_features = []
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window]
            hog_features.append(hog_feat1)
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window]
            hog_features.append(hog_feat2)
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window]
            hog_features.append(hog_feat3)
            hog_features=np.ravel(hog_features)


            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
            # Get color features
            spatial_features = hist.bin_spatial(subimg, size=spatial_size)

            hist_features = hist.color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            features = []
            features.append(spatial_features)
            features.append(hist_features)
            features.append(hog_features)
            features=np.concatenate(features)
            test_features = X_scaler.transform(features.reshape(1, -1))
            test_prediction = svc.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)+xstart
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxes.append(((int(xbox_left), int(ytop_draw+ystart)),(int(xbox_left+win_draw),int(ytop_draw+win_draw+ystart))))
    return boxes
```
which gets such params as image,searching region and scale. 
I used different scale and x,y parameters, with splitting image and using different scaling for each part but it increases the difference and my computer works slowly on video processing. 
My best optimum result was achieved with this parameters:
```python
boxes = find_cars(draw_image, 360, 650,559,1280, 1.5)
```

after finding boxes there are some processing methods calculating heatmap, very powerful instrument with labels function, which allows find cars
From the positive detections I created a heatmap and then thresholded that map =1 to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I t


:
```python
def hog_subsampling_window_search_tuned(image):
    image = mpimg.imread(image_p)
    draw_image = np.copy(image)
    boxes = []

    boxes = find_cars(draw_image, 360, 650,559,1280, 1.5)


    window_img = sw.draw_boxes(draw_image, boxes, color=(0, 0, 255), thick=6)
    heat = np.zeros_like(window_img[:, :, 0]).astype(np.float)
    heatmap = psg.add_heat(heat, boxes)
    heatmap = psg.apply_threshold(heatmap, 1)
    labels = label(heatmap)
    print(labels[1], 'cars found')
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(24, 9))
    ax0.set_title("no filtering", fontsize=15)
    ax0.imshow(window_img)
    ax1.set_title("after labeling", fontsize=15)
    ax1.imshow(psg.draw_labeled_bboxes(draw_image, labels))
    ax2.set_title("heat map", fontsize=15)
    ax2.imshow(heatmap, cmap='gray')

    plt.show()
```
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]


as we see all cars are detected correctly


---

### Video Implementation

For video implementation I've used my previous functions and also I've created detected cars model which saves 12 latest founded boxes collections
```python
from collections import deque
class Vehicle_Detect():
    def __init__(self):
        # history of rectangles previous n frames
        self.prev_boxes = deque(maxlen=12)
```
function process video: 

```python
def process_video(road_image):
    draw_image = np.copy(road_image)

    boxes = []

    boxes = find_cars(draw_image, 350, 650, 559, 1280, 1.5)


    if len(boxes) > 0:
        detected_vehicles.prev_boxes.append(boxes)

    window_img = sw.draw_boxes(draw_image, boxes, color=(0, 0, 255), thick=6)
    heatmap = np.zeros_like(window_img[:, :, 0]).astype(np.float)
    for rect in detected_vehicles.prev_boxes:
        heatmap = psg.add_heat(heatmap, rect)
    heatmap = psg.apply_threshold(heatmap, 2+len(detected_vehicles.prev_boxes)//2)
    labels = label(heatmap)
    result = psg.draw_labeled_bboxes(draw_image, labels)
    return  result

```
At first we use find cars method, then if length of founded boxes>0 we add it to model. 
After combining heatmap and labels method we draw boxes using all previous founded boxes for cars and as thresholds I use 2+len(detected_vehicles.prev_boxes)//2

result video is on the folder.




---


