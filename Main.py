import Data as data
import cv2
import matplotlib.pyplot as plt
import numpy as np
import Features as ft
import time
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import SlidingWindow as sw
import matplotlib.image as mpimg
import Histogram as hist
import pickle
import Processing as psg
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from Vehicle import Vehicle_Detect
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
filename = 'finalized_model.p'
try:
    loaded_model = pickle.load(open(filename, 'rb'))
    svc = loaded_model["svc"]
    X_scaler = loaded_model["X-scaler"]
    orient = loaded_model["orient"]
    pix_per_cell = loaded_model["pix_per_cell"]
    cell_per_block = loaded_model["cell_per_block"]
    spatial_size = loaded_model["spatial_size"]
    hist_bins = loaded_model["hist_bins"]
    color_space = loaded_model["color_space"]

except Exception as e:
    cars, notcars = data.getDataset()
    # Define parameters for feature extraction
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block

    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins

    # Extracting features
    car_features = ft.extract_features(cars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
    print('Car samples: ', len(car_features))
    notcar_features = ft.extract_features(notcars, color_space=color_space,
                                          spatial_size=spatial_size, hist_bins=hist_bins,
                                          orient=orient, pix_per_cell=pix_per_cell,
                                          cell_per_block=cell_per_block,
                                          hog_channel=hog_channel, spatial_feat=spatial_feat,
                                          hist_feat=hist_feat, hog_feat=hog_feat)
    print('Notcar samples: ', len(notcar_features))
    # Normalizing features
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    X_scaler = StandardScaler().fit(X)  # Fit a per-column scaler
    scaled_X = X_scaler.transform(X)  # Apply the scaler to X

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))  # Define the labels vector
    rand_state = np.random.randint(0, 100)
    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
    print('Using spatial binning of:', spatial_size,
          'and', hist_bins, 'histogram bins')
    print('Using HOG:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
    # save the model to disk
    t = time.time()  # Start time
    model = {"svc": svc, "X-scaler": X_scaler, "orient": orient, "pix_per_cell": pix_per_cell,
             "cell_per_block": cell_per_block, "spatial_size": spatial_size, "hist_bins": hist_bins,"color_space":color_space}
    pickle.dump(model, open(filename, 'wb'))
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


detected_vehicles=Vehicle_Detect()
for image_p in glob.glob('test_images/test*.jpg'):
    image=mpimg.imread(image_p)
    hog_subsampling_window_search_tuned(image)

# test_out_file2 = 'test_video_out_2.mp4'
# clip_test2 = VideoFileClip('project.avi')
# clip_test_out2 = clip_test2.fl_image(process_video)
# clip_test_out2.write_videofile(test_out_file2, audio=False)
# HTML("""
# <video width="960" height="540" controls>
#   <source src="{0}">
# </video>
# """.format(test_out_file2))

