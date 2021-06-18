# Repository for PhD thesis 'Understanding landscape change in support of opium monitoring in Afghanistan'
A copy of the thesis is available at: (TBC)
Data access statement is available at: https://doi.org/10.17862/cranfield.rd.14447400 

## Training with sparsely labelled datasets
The script in *Sparse FCN-8* is an adaptation of the FCN-8 code implemented in https://github.com/dspix/deepjet for densely labelled datasets (TensorFlow v1). The generalised FCN-8 model trained on agricultural masks from 2007, 2008, 2009, 2015, 2016 and 2017 is available at https://doi.org/10.17862/cranfield.rd.14447400, but 3-band level-1A input imagery (NIR, R, G) requires Iteratively Reweighted Multivariate Alteration Detection (IR-MAD) normalistion (https://github.com/mortcanty/CRCPython/tree/master/src/CHAPTER9) to the DMC image (du000aa3t) acquired on 27 April 2007 across Helmand and Kandahar.

## Agricultural land classification
The script in *FCN-8 prediction* is used for classifying agricultural land with 3-band input imagery (NIR, R, G) and an FCN-8 model. The calibration coefficients for an example Level-1A Sentinel-2 tile (L1C_T41SPR_A019765_20190405T061721) is provided in *FCN-8 prediction* to replicate a subset of the agricultural land classification.

## Localised intersection over union
The script in *Localised IoU* is used for calculating the intersection over union for each block of agriculture from the reference agricultural mask and the classified agricultural mask to quantify the localised differences in agricultural mapping.
