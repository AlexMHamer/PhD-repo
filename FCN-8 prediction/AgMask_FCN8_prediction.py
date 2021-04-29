import numpy as np
import tensorflow as tf
from osgeo import gdal
import glob

def AgMask_predict(image, model_file, checkpoint_path):
    for i in image:
        image = gdal.Open(i)
        nir, red, grn = image.ReadAsArray() # 3 band image
        arr = np.stack((nir, red, grn))
        bands, rows, cols = arr.shape
        data = arr.transpose(1,2,0)
        graph = tf.Graph()
        config = tf.ConfigProto(device_count = {'GPU': 0})
        with graph.as_default():
            with tf.Session(config=config) as sess:
                model_import = tf.train.import_meta_graph(model_file)
                model_import.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
                x_placeholder = graph.get_tensor_by_name('Placeholder:0')
                y_placeholder = graph.get_tensor_by_name('Placeholder_1:0')
                prediction = graph.get_tensor_by_name('upscale_orig/upscale_orig:0')
                out_pred = sess.run(prediction, feed_dict={x_placeholder: [data]})

        ag_threshold = np.argmax(out_pred, -1) # Ag mask creation
        ag_threshold = ag_threshold.reshape(rows, cols)

        trans = image.GetGeoTransform()
        proj = image.GetProjection()
        outdriver = gdal.GetDriverByName('GTIFF')
        outdata   = outdriver.Create(i[:-4] + '_pred.tif', cols, rows, 1, gdal.GDT_Float32)
        outdata.GetRasterBand(1).WriteArray(ag_threshold)
        outdata.SetGeoTransform(trans)
        outdata.SetProjection(proj)

# Function
inputImage = glob.glob('*.tif')
model = 'FCN8_model.ckpt.meta'
checkpoint = './Model/'

AgMask_predict(inputImage, model, checkpoint)
