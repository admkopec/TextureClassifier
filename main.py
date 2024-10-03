import os
import joblib
import numpy as np
import cv2
from skimage.io import imread
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

display = True

def prepareImages(src, pklname):
    # load images from path, calculate features, write them as arrays to a dictionary,
    # together with labels and metadata. The dictionary is written to a pickle file
    # named '{pklname}.pkl'.

    data = dict()
    data['description'] = f'texture {pklname} images in rgb'
    data['label'] = []
    data['filename'] = []
    data['featuresGLCM'] = []
    data['featuresLBP'] = []
    data['featuresCombined'] = []

    pklname = f"{pklname}.pkl"

    # read all images in PATH, resize and write to DESTINATION_PATH
    for label in os.listdir(src):
        current_path = os.path.join(src, label)
        if os.path.isdir(current_path):
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    image = imread(os.path.join(current_path, file))
                    glcmFeatures = calculateGlcmFeatures(image)
                    lbpFeatures = calculateLbpFeatures(image)
                    combinedFeatures = np.concatenate((glcmFeatures, lbpFeatures))
                    data['label'].append(label)
                    data['filename'].append(file)
                    data['featuresGLCM'].append(glcmFeatures)
                    data['featuresLBP'].append(lbpFeatures)
                    data['featuresCombined'].append(combinedFeatures)

            joblib.dump(data, pklname)


def calculateGlcmFeatures(image):
    stats = ['r_energy', 'r_correlation', 'r_contrast', 'r_homogeneity', 'g_energy', 'g_correlation', 'g_contrast',
             'g_homogeneity', 'b_energy', 'b_correlation', 'b_contrast', 'b_homogeneity', 'h_energy', 'h_correlation',
             'h_contrast', 'h_homogeneity', 's_energy', 's_correlation', 's_contrast', 's_homogeneity', 'v_energy',
             'v_correlation', 'v_contrast', 'v_homogeneity']
    offsetdist = [1, 1, 2, 1]
    offsetang = [7 * np.pi / 4, np.pi / 2, np.pi / 4, np.pi / 6]
    features = np.zeros(len(stats))

    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    glcm_dict = {
        'r': graycomatrix(image[..., 0], distances=offsetdist, angles=offsetang, levels=256, symmetric=False,
                          normed=True),
        'g': graycomatrix(image[..., 1], distances=offsetdist, angles=offsetang, levels=256, symmetric=False,
                          normed=True),
        'b': graycomatrix(image[..., 2], distances=offsetdist, angles=offsetang, levels=256, symmetric=False,
                          normed=True),
        'h': graycomatrix(image_hsv[..., 0], distances=offsetdist, angles=offsetang, levels=256, symmetric=False,
                          normed=True),
        's': graycomatrix(image_hsv[..., 1], distances=offsetdist, angles=offsetang, levels=256, symmetric=False,
                          normed=True),
        'v': graycomatrix(image_hsv[..., 2], distances=offsetdist, angles=offsetang, levels=256, symmetric=False,
                          normed=True)
    }

    for idx, stat in enumerate(stats):
        channel = stat[0]
        channel_glcm = glcm_dict[channel]
        feat_val = graycoprops(channel_glcm, stat[2::])[0, 0]
        features[idx] = feat_val

    return features


def calculateLbpFeatures(image):
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    numberOfPoints = 30
    radius = 7
    e = 1e-7

    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lbp = local_binary_pattern(image_gray, numberOfPoints, radius, method="uniform")

    (features, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, numberOfPoints + 3),
                                 range=(0, numberOfPoints + 2))
    # normalize the histogram
    features = features.astype("float")
    features /= (features.sum() + e)

    return features

import cv2
import os
import numpy as np
from skimage import feature
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure

img = imread('./textures/train/KTH_aluminium_foil/1.jpg')

resized_img = resize(img, (128*4, 64*4))
print(resized_img.shape)
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis=-1)

exit(0)

prepareImages('textures/train', 'train')
prepareImages('textures/valid', 'test')

trainSet = joblib.load('train.pkl')
testSet = joblib.load('test.pkl')

print('Train set:')
print('number of samples: ', len(trainSet['featuresCombined']))
print('keys: ', list(trainSet.keys()))
print('description: ', trainSet['description'])
print('labels:', np.unique(trainSet['label']))

print('---------------------------------------')

print('Test set:')
print('number of samples: ', len(testSet['featuresCombined']))
print('keys: ', list(testSet.keys()))
print('description: ', testSet['description'])
print('labels:', np.unique(testSet['label']))

for featureSet in ['featuresGLCM', 'featuresLBP', 'featuresCombined']:
    print('---------------------------------------')
    print(f'Using the {featureSet} feature set')
    # Train classifier on the train set
    classifier = RandomForestClassifier()
    classifier.fit(trainSet[featureSet], trainSet['label'])

    # Validation set predictions
    pred_val_labels = classifier.predict(testSet[featureSet])
    raw_probabilities = classifier.predict_proba(testSet[featureSet])[:, 1]

    # Score computation
    score_val = round(classifier.score(testSet[featureSet], testSet['label'])*100, 2)
    print(f"Score: {score_val}%")

    bal_acc_val = round(balanced_accuracy_score(testSet['label'], pred_val_labels)*100, 2)
    print(f"Balanced accuracy: {bal_acc_val}%")

    if not display or featureSet != 'featuresCombined':
        continue

    for idx in range(0, len(pred_val_labels)):
        if pred_val_labels[idx] == testSet['label'][idx] and raw_probabilities[idx] >= 0.8:
            print(f"Showing a good classification of {pred_val_labels[idx]} with probability: {raw_probabilities[idx]}")
            cv2.imshow(testSet['filename'][idx], cv2.imread(os.path.join(os.path.join('textures/valid', testSet['label'][idx]), testSet['filename'][idx])))
            cv2.waitKey()
            cv2.destroyAllWindows()

        if pred_val_labels[idx] != testSet['label'][idx] and raw_probabilities[idx] >= 0.2:
            print(f"Showing a bad classification of {pred_val_labels[idx]} with probability: {raw_probabilities[idx]} instead of {testSet['label'][idx]}")
            cv2.imshow(testSet['filename'][idx], cv2.imread(os.path.join(os.path.join('textures/valid', testSet['label'][idx]), testSet['filename'][idx])))
            cv2.waitKey()
            cv2.destroyAllWindows()
