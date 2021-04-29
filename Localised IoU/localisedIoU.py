import gdal
import numpy as np

from skimage.measure import label, regionprops
from sklearn.metrics import jaccard_score

plot_labels_img = gdal.Open('reference.tif')
plot_labels = plot_labels_img.ReadAsArray()
plot_labels = plot_labels[0]
rows, cols = plot_labels.shape

plot_prediction_img = gdal.Open('prediction.tif')
plot_prediction = plot_prediction_img.ReadAsArray()
print(plot_labels.shape)
print(plot_prediction.shape)

plot_labels[plot_labels < 0] = 0
classes,count = np.array(np.unique(plot_labels, return_counts=True))

# label image regions
label_image = label(plot_labels)

region_coordinates = []
region_cutOuts_labels = []
region_cutOuts_pred = []
rectangles = []

for region in regionprops(label_image):
    region_coordinates.append(region.bbox)
    if region.area >= 1: # Take regions with greater than or equal to 1 pixel
        minr, minc, maxr, maxc = region.bbox
        region_cut_labels_ref = plot_labels[minr:maxr, minc:maxc]
        region_cutOuts_labels.append(region_cut_labels_ref)

        region_cut_labels_pred = plot_prediction[minr:maxr, minc:maxc]
        region_cutOuts_pred.append(region_cut_labels_pred)


region_cutOuts_labels = np.array(region_cutOuts_labels)
region_cutOuts_pred = np.array(region_cutOuts_pred)
print(region_cutOuts_labels.shape)
print(region_cutOuts_pred.shape)

# Calculate IoU
labsPred_IoU = np.stack([region_cutOuts_labels, region_cutOuts_pred], axis=-1)
print(labsPred_IoU.shape)

IoU_object = []

for i in labsPred_IoU:
    reference = i[0]
    prediction = i[1]
    IoU = jaccard_score(reference, prediction, average=None)
    IoU_object.append(IoU[1]) # Selects agriculture class

IoU_object = np.array(IoU_object)

newValues = []

# Bin the data into 10 bins
for i in IoU_object:
    if 0 <= i < 0.1:
        newValues.append(1)
    elif 0.1 <= i < 0.2:
        newValues.append(2)
    elif 0.2 <= i < 0.3:
        newValues.append(3)
    elif 0.3 <= i < 0.4:
        newValues.append(4)
    elif 0.4 <= i < 0.5:
        newValues.append(5)
    elif 0.5 <= i < 0.6:
        newValues.append(6)
    elif 0.6 <= i < 0.7:
        newValues.append(7)
    elif 0.7 <= i < 0.8:
        newValues.append(8)
    elif 0.8 <= i < 0.9:
        newValues.append(9)
    elif 0.9 <= i <= 1.0:
        newValues.append(10)

newValues = np.array(newValues)

label_image_flat = label_image.flatten()
label_image_IoU = []
for i in label_image_flat:
    value = i-1
    if value > -1:
        label_image_IoU.append(newValues[value])
    else:
        label_image_IoU.append(0)

classes,count = np.array(np.unique(np.array(label_image_IoU), return_counts=True))

ag_threshold = np.array(label_image_IoU).reshape(rows, cols)
trans = plot_labels_img.GetGeoTransform()
proj = plot_labels_img.GetProjection()
outdriver = gdal.GetDriverByName('GTIFF')
outdata   = outdriver.Create('localisedIoU.tif', cols, rows, 1, gdal.GDT_Float32) # Output file
outdata.GetRasterBand(1).WriteArray(ag_threshold)
outdata.SetGeoTransform(trans)
outdata.SetProjection(proj)
