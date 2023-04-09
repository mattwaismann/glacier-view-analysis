import tensorflow as tf
import numpy as np
import landsat_bands

def replace_min_with_second_min(raster_band):
    first_min = raster_band.min()
    sorted_intensities = sorted(set(raster_band.flatten()))
    if len(sorted_intensities) == 1: 
        return raster_band
    else:
        second_min = sorted(set(raster_band.flatten()))[1]
    return np.where(raster_band == first_min, second_min, raster_band)

def normalize_rasters(rasters):
    normalized_rasters = []
    band_index = 2
    for raster in rasters:
        for band_number in range(raster.shape[band_index]):
            raster_band = raster[:,:,band_number]
            raster_band = replace_min_with_second_min(raster_band)
            temp_max = raster_band.max()
            temp_min = raster_band.min()
            if temp_max == temp_min: 
                normalized_raster_band = raster_band
            else:
                normalized_raster_band = (raster_band-temp_min)/(temp_max-temp_min)
            raster[:,:,band_number] = normalized_raster_band
        normalized_rasters.append(raster)
    return normalized_rasters 

def resize_rasters(rasters,dim):
    resized_rasters = []
    for raster in rasters:
        resized_rasters.append(tf.image.resize(raster, dim).numpy())
    return resized_rasters

def get_common_bands_from_list_of_numpy_arrays(list_of_numpy_arrays, bands_list, list_of_identifying_strings=None):
    is_l5 = True
    is_l7 = True
    is_l8 = True
    for band in bands_list:
        if band not in landsat_bands.l5_bands:
            is_l5 = False
        if band not in landsat_bands.l7_bands:
            is_l7 = False
        if band not in landsat_bands.l8_bands:
            is_l8 = False
    if is_l5:
        l5_sub = [landsat_bands.l5_bands[band] for band in bands_list]
    if is_l7:
        l7_sub = [landsat_bands.l7_bands[band] for band in bands_list]
    if is_l8:
        l8_sub = [landsat_bands.l8_bands[band] for band in bands_list]
    common_arrays = []
    image_file_names = []
    for i in range(len(list_of_numpy_arrays)):
        n_bands = list_of_numpy_arrays[i].shape[2]
        if n_bands == 8 and is_l5:
            new_array = list_of_numpy_arrays[i][:,:,l5_sub]
            if list_of_identifying_strings: file_name = list_of_identifying_strings[i]
        if n_bands == 10 and is_l7:
            new_array = list_of_numpy_arrays[i][:,:,l7_sub]
            if list_of_identifying_strings: file_name = list_of_identifying_strings[i]
        if n_bands == 12 and is_l8:
            new_array = list_of_numpy_arrays[i][:,:,l8_sub]
            if list_of_identifying_strings: file_name = list_of_identifying_strings[i]
        common_arrays.append(new_array)
        if list_of_identifying_strings: 
            image_file_names.append(file_name)
    if list_of_identifying_strings: return (common_arrays,image_file_names)
    else: return common_arrays
    

def subset_on_labels(X_train,y_train, labels, oos_ind):
    X_train_sub = X_train[labels,:,:,:]
    y_train_sub = y_train[labels,:,:,:]
    match_inds = []
    for s in range(len(oos_ind)):
        for idx,b in enumerate(range(len(labels))):
            if oos_ind[s] == labels[b]:
                match_inds.append(idx)
    return X_train_sub,y_train_sub, match_inds

def threshold(predictions, thresh):
    preds = predictions.copy()
    preds[preds>thresh] = 1
    preds[preds<thresh] = 0
    return preds

def clean_ts(time_series, dates, glacier_id):
    '''
    time_series: (4d array (time,dim1,dim2,channel))
    dates: (date array) a 1d array of dates
    model: (keras model)
    glacier_id (string) 
    
    '''


    #search middle of image for a 0, if you find one then there is stripes
    ts = time_series.copy()
    dates_ts = dates.copy()
    no_stripes_ind = []
    for t in range(ts.shape[0]):
        if (ts[t,20:100,20:100,0] == 0.01).any():
            pass
        else:
            no_stripes_ind.append(t)
    
    ts = ts[no_stripes_ind]
    dates_ts = dates_ts[no_stripes_ind]
    
    #pred_ts = model.predict(ts)
    #pred_ts_mask = threshold(pred_ts, 0.5)
    
    #sse = []
    #good_mask_example = mask_dict[glacier_id][:,:,0]
    #good_mask_area = np.sum(good_mask_example)
    #for i in range(pred_ts_mask.shape[0]):
        #other = pred_ts_mask[i,:,:,0]
        #sse.append(np.sum((good_mask_example-other)**2))
    #keep_ind = np.where(np.array(sse) < good_mask_area/1)
    #ts = ts[keep_ind]
    #dates_ts = dates_ts[keep_ind]
    return ts, dates_ts