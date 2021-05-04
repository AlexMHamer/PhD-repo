# Repository for PhD thesis 'Understanding landscape change in support of opium monitoring in Afghanistan'
Data access statement is available at: https://doi.org/10.17862/cranfield.rd.14447400 

# Code for training the FCN-8 with sparsely labelled datasets
The code in 'Sparse FCN-8' is an adaptation of the FCN-8 code implemented in https://github.com/dspix/deepjet for densely labelled datasets (TensorFlow v1). The generalised FCN-8 model trained on agricultural masks from 2007, 2008, 2009, 2015, 2016 and 2017 is made available through the data access statement, but 3-band input imagery (NIR, R, G) requires Iteratively Reweighted Multivariate Alteration Detection (IR-MAD) normalistion (https://github.com/mortcanty/CRCPython/tree/master/src/CHAPTER9) to the DMC image (du000aa3t) acquired on 27 April 2007 across Helmand and Kandahar.

# Code for predicting with the FCN-8 model
The code in 'FCN-8 prediction' is used for classifying agricultural land with 3-band input imagery (NIR, R, G) and the FCN-8 model.

# Code for calculating localised intersection over union
The code in 'Localised IoU' is used for calculating the intersection over union for each block of agriculture from the reference agricultural mask and the classified agricultural mask.
