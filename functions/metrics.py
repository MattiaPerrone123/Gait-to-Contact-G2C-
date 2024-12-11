import scipy.stats as stats
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.measure import label, regionprops
from sklearn.metrics import normalized_mutual_info_score


import scipy.stats as stats
import numpy as np
from skimage.measure import label, regionprops


import numpy as np
from skimage.measure import label, regionprops

def count_blobs(image):
    """Count the number of blobs (connected components) in a binary image"""
    labeled_image, num_blobs=label(image, background=0, return_num=True)
    return num_blobs

def keep_two_largest_blobs(image):
    """Retain only the two largest blobs in a binary image"""
    labeled_image=label(image)
    regions=regionprops(labeled_image)
    if len(regions) <= 2:
        return image
    largest_regions=sorted(regions, key=lambda x: x.area, reverse=True)[:2]
    mask=np.zeros_like(image)
    for region in largest_regions:
        mask[labeled_image==region.label]=1
    return mask

def calculate_extent(image, thresh_zero):
    """Calculate the x and y extents of non-zero pixels in a binary image"""
    y_coords, x_coords=np.where(image > thresh_zero)
    if len(x_coords) == 0 or len(y_coords) == 0:
        raise ValueError("No pixels above the threshold in the image")
    x_extent=np.max(x_coords) - np.min(x_coords)
    y_extent=np.max(y_coords) - np.min(y_coords)
    return x_extent, y_extent

def postprocess_images_cont(predictions_test, thresh_zero):
    """Postprocess predictions by splitting into two components and calculating extents"""
    x_ext_first_list, y_ext_first_list, x_ext_second_list, y_ext_second_list=[], [], [], []
    for curr_img in predictions_test:
        curr_img[curr_img > thresh_zero]=1
        first_component=curr_img[:, :curr_img.shape[-1]//2]
        x_ext_first, y_ext_first=calculate_extent(first_component, thresh_zero)
        x_ext_first_list.append(x_ext_first)
        y_ext_first_list.append(y_ext_first)
        second_component=curr_img[:, curr_img.shape[-1]//2:]
        x_ext_second, y_ext_second=calculate_extent(second_component, thresh_zero)
        x_ext_second_list.append(x_ext_second)
        y_ext_second_list.append(y_ext_second)
    return x_ext_first_list, y_ext_first_list, x_ext_second_list, y_ext_second_list

def calculate_area_cont(image):
    """Calculate non-weighted and weighted areas for two components of an image"""
    curr_img=image[0].copy()
    first_component=curr_img[:, :curr_img.shape[-1]//2]
    second_component=curr_img[:, curr_img.shape[-1]//2:]
    non_weighted_area_first=np.sum(first_component > 0)
    non_weighted_area_second=np.sum(second_component > 0)
    weighted_area_first=np.sum(first_component)
    weighted_area_second=np.sum(second_component)
    return non_weighted_area_first, non_weighted_area_second, weighted_area_first, weighted_area_second

def calculate_weighted_centroid(component, grid):
    """Calculate the weighted centroid of an image component"""
    X, Y=grid
    total_intensity=component.sum()
    if total_intensity == 0:
        return 0, 0
    x_center=(X * component).sum() / total_intensity
    y_center=(Y * component).sum() / total_intensity
    return x_center, y_center

def calculate_non_weighted_centroid(component):
    """Calculate the non-weighted centroid of a binary image component"""
    binary_component=(component > 0).astype(int)
    total_pixels=binary_component.sum()
    if total_pixels == 0:
        return 0, 0
    Y, X=np.ogrid[:binary_component.shape[0], :binary_component.shape[1]]
    x_center=(X * binary_component).sum() / total_pixels
    y_center=(Y * binary_component).sum() / total_pixels
    return x_center, y_center

def calculate_centroids_cont(image):
    """Calculate weighted and non-weighted centroids for two components of an image"""
    curr_img=image[0].copy()
    mid_point=curr_img.shape[-1] // 2
    first_component=curr_img[:, :mid_point]
    second_component=curr_img[:, mid_point:]
    Y_first, X_first=np.ogrid[:first_component.shape[0], :first_component.shape[1]]
    Y_second, X_second=np.ogrid[:second_component.shape[0], :second_component.shape[1]]
    x_center_first_w, y_center_first_w=calculate_weighted_centroid(first_component, (X_first, Y_first))
    x_center_second_w, y_center_second_w=calculate_weighted_centroid(second_component, (X_second, Y_second))
    x_center_first_nw, y_center_first_nw=calculate_non_weighted_centroid(first_component)
    x_center_second_nw, y_center_second_nw=calculate_non_weighted_centroid(second_component)
    return (int(x_center_first_w), int(y_center_first_w)), (int(x_center_second_w), int(y_center_second_w)), \
           (int(x_center_first_nw), int(y_center_first_nw)), (int(x_center_second_nw), int(y_center_second_nw))




def angle_with_horizontal(P1, P2):
    """Calculate the angle between a line segment and the horizontal axis"""
    if P1[1] > P2[1]:
        P1, P2=P2, P1
    dx=P2[0] - P1[0]
    dy=P2[1] - P1[1]
    theta_radians=np.arctan2(dy, dx)
    return np.degrees(theta_radians)

def calculate_nrmse(ground_truth, predictions):
    """Calculate the normalized root mean squared error (NRMSE)"""
    ground_truth=np.array(ground_truth) if isinstance(ground_truth, list) else ground_truth
    predictions=np.array(predictions) if isinstance(predictions, list) else predictions
    if ground_truth.shape != predictions.shape:
        raise ValueError("Shapes of ground truth and predictions must match")
    mse=np.mean((ground_truth - predictions)**2)
    rmse=np.sqrt(mse)
    value_range=np.max(ground_truth) - np.min(ground_truth)
    if value_range == 0:
        raise ValueError("Ground truth values have no variation")
    return rmse / value_range

def calculate_mape(actual_values, predicted_values, epsilon=1e-6):
    """Calculate the Mean Absolute Percentage Error (MAPE)"""
    actual_values=np.array(actual_values)
    predicted_values=np.array(predicted_values)
    adjusted_denominator=np.where(actual_values == 0, np.maximum(actual_values, epsilon), actual_values)
    errors=np.abs((actual_values - predicted_values) / adjusted_denominator)
    return np.mean(errors) * 100

def distance_centroids(point1, point2, width, height):
    """Calculate the normalized Euclidean distance between two points"""
    diagonal=np.sqrt(width**2 + height**2)
    distance=np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    return (distance / diagonal) * 100

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate the confidence interval for the mean of a dataset"""
    data=np.array(data)
    mean=np.mean(data)
    stderr=stats.sem(data)
    margin=stderr * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return mean - margin, mean + margin

def calculate_blob_extents(predictions, ground_truth, thresh_zero):
    """Calculate blob extents for predictions and ground truth"""
    x_ext_pred, y_ext_pred, x_ext_ground, y_ext_ground={}, {}, {}, {}
    x_ext_pred['first'], y_ext_pred['first'], x_ext_pred['second'], y_ext_pred['second']=\
        postprocess_images_cont(predictions[:, 0], thresh_zero)
    x_ext_ground['first'], y_ext_ground['first'], x_ext_ground['second'], y_ext_ground['second']=\
        postprocess_images_cont(ground_truth[:, 0], thresh_zero)
    return x_ext_pred, y_ext_pred, x_ext_ground, y_ext_ground

def calculate_areas(predictions, ground_truth):
    """Calculate weighted and non-weighted areas for predictions and ground truth"""
    areas_pred={'non_weighted_first': [], 'weighted_first': [], 'non_weighted_second': [], 'weighted_second': []}
    areas_ground={'non_weighted_first': [], 'weighted_first': [], 'non_weighted_second': [], 'weighted_second': []}
    for pred in predictions:
        areas=calculate_area_cont(pred)
        for key, value in zip(areas_pred.keys(), areas):
            areas_pred[key].append(value)
    for ground in ground_truth:
        areas=calculate_area_cont(ground)
        for key, value in zip(areas_ground.keys(), areas):
            areas_ground[key].append(value)
    return areas_pred, areas_ground

def compute_centroids(predictions, ground_truth):
    """Compute centroids for predictions and ground truth"""
    centroids_pred={'non_weighted_first': [], 'weighted_first': [], 'non_weighted_second': [], 'weighted_second': []}
    centroids_ground={'non_weighted_first': [], 'weighted_first': [], 'non_weighted_second': [], 'weighted_second': []}
    for pred in predictions:
        non_w_first, non_w_second, w_first, w_second=calculate_centroids_cont(pred)
        centroids_pred['non_weighted_first'].append(non_w_first)
        centroids_pred['non_weighted_second'].append(non_w_second)
        centroids_pred['weighted_first'].append(w_first)
        centroids_pred['weighted_second'].append(w_second)
    for ground in ground_truth:
        non_w_first, non_w_second, w_first, w_second=calculate_centroids_cont(ground)
        centroids_ground['non_weighted_first'].append(non_w_first)
        centroids_ground['non_weighted_second'].append(non_w_second)
        centroids_ground['weighted_first'].append(w_first)
        centroids_ground['weighted_second'].append(w_second)
    return centroids_pred, centroids_ground

def compute_centroid_distances(centroids_pred, centroids_ground, image_shape):
    """Compute distances between centroids for predictions and ground truth"""
    distances={'weighted_first': [], 'non_weighted_first': [], 'weighted_second': [], 'non_weighted_second': []}
    for i in range(len(centroids_pred['weighted_first'])):
        for key in distances.keys():
            distances[key].append(
                distance_centroids(centroids_ground[key][i], centroids_pred[key][i], image_shape[1], image_shape[0])
            )
    return distances

def compute_angles(centroids_pred, centroids_ground, image_shape):
    """Compute angles between centroids for predictions and ground truth"""
    angles={'weighted': {'ground': [], 'pred': []}, 'non_weighted': {'ground': [], 'pred': []}}
    offset=int(image_shape[1] / 2)
    adjusted_ground_second=[(x[0] + offset, x[1]) for x in centroids_ground['weighted_second']]
    adjusted_pred_second=[(x[0] + offset, x[1]) for x in centroids_pred['weighted_second']]
    for i in range(len(adjusted_ground_second)):
        angles['weighted']['ground'].append(
            angle_with_horizontal(adjusted_ground_second[i], centroids_ground['weighted_first'][i])
        )
        angles['non_weighted']['ground'].append(
            angle_with_horizontal(centroids_ground['non_weighted_second'][i], centroids_ground['non_weighted_first'][i])
        )
        angles['weighted']['pred'].append(
            angle_with_horizontal(adjusted_pred_second[i], centroids_pred['weighted_first'][i])
        )
        angles['non_weighted']['pred'].append(
            angle_with_horizontal(centroids_pred['non_weighted_second'][i], centroids_pred['non_weighted_first'][i])
        )
    return angles



def calculate_ape(ground_truth, predictions):
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    return np.abs((ground_truth - predictions) / ground_truth) * 100


def sd_ape(ape_array, mape):
    return np.sqrt(np.sum(np.square(ape_array-mape)))/len(ape_array)
    


def calculate_ape(ground_truth, predictions):
    """Calculate the Absolute Percentage Error (APE)"""
    ground_truth=np.array(ground_truth)
    predictions=np.array(predictions)
    return np.abs((ground_truth - predictions) / ground_truth) * 100



def sd_ape(ape_array, mape):
    """Calculate the standard deviation of the Absolute Percentage Error (APE)"""
    return np.sqrt(np.sum(np.square(ape_array - mape))) / len(ape_array)



def calculate_nmi(predictions, ground_truth, region=None):
    """Calculate the Normalized Mutual Information (NMI) for predictions and ground truth"""
    nmi_scores=[]
    for i in range(len(predictions)):
        if region:
            y_start, y_end, x_start, x_end=region
            pred_region=predictions[i, 0, y_start:y_end, x_start:x_end]
            gt_region=ground_truth[i, 0, y_start:y_end, x_start:x_end]
        else:
            pred_region=predictions[i, 0]
            gt_region=ground_truth[i, 0]
        nmi=normalized_mutual_info_score(pred_region.flatten(), gt_region.flatten())
        nmi_scores.append(nmi)
    return nmi_scores



def calculate_ssim(predictions, ground_truth, region=None):
    """Calculate the Structural Similarity (SSIM) for predictions and ground truth"""
    ssim_scores=[]
    for i in range(len(predictions)):
        if region:
            y_start, y_end, x_start, x_end=region
            pred_region=predictions[i, 0, y_start:y_end, x_start:x_end]
            gt_region=ground_truth[i, 0, y_start:y_end, x_start:x_end]
        else:
            pred_region=predictions[i, 0]
            gt_region=ground_truth[i, 0]
        ssim_score=ssim(pred_region, gt_region, data_range=gt_region.max() - gt_region.min())
        ssim_scores.append(ssim_score)
    return ssim_scores



def determine_dynamic_roi(predictions, thresh_zero):
    """Determine the dynamic region of interest (ROI) based on blob boundaries"""
    blobs_x_ext, blobs_y_ext, _, _=postprocess_images_cont(predictions[:, 0], thresh_zero)
    y_start=int(min(blobs_y_ext))
    y_end=int(max(blobs_y_ext))
    x_start=int(min(blobs_x_ext))
    x_end=int(max(blobs_x_ext))
    return y_start, y_end, x_start, x_end
