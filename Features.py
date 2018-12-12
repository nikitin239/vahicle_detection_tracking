import Histogram as hist
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# Define a function to extract features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file_p in imgs:
        file_features = []
        image = mpimg.imread(file_p) # Read in each imageone by one
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(hist.get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))

            hog_features = np.ravel(hog_features)
        else:
            img_copy=feature_image[:, :, hog_channel]
            hog_features = hist.get_hog_features(img_copy, orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            # Apply bin_spatial() to get spatial color features
        spatial_features = hist.bin_spatial(feature_image, size=spatial_size)
            # Apply color_hist() also with a color space option now
        hist_features = hist.color_hist(feature_image, nbins=hist_bins)
            # Append the new feature vector to the features list



        features.append(np.concatenate((spatial_features, hist_features,hog_features)))
        #feature_image=cv2.flip(feature_image,1) # Augment the dataset with flipped images
        # file_features = img_features(feature_image, spatial_feat,spatial_size, hist_feat, hog_feat, hist_bins, orient,
        #                 pix_per_cell, cell_per_block, hog_channel)
        # features.append(np.concatenate(file_features))
    return features # Return list of feature vectors


