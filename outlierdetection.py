import numpy as np
from collections import Counter

###################################################################################################################

# Detecting outliers in the data - Tukey's Method for identfying outliers: An outlier step is calculated as 1.5 times
# the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR
# for that feature is considered abnormal.
# http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/

def outlier_detection(data, features, threshold=1.5):
    if threshold < 0:
        return data

    outliers  = []
    # For each feature find the data points with extreme high or low values
    for feature in features:

        # Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.percentile(data[feature], 25)

        # Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.percentile(data[feature], 75)

        # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = threshold * (Q3 - Q1)

        # Display the outliers
        #print ("Data points considered outliers for the feature '{}':".format(feature))
        feature_outliers = data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))]
        #display(feature_outliers)

        # Add the outliers indices to the list of outliers for removal later on
        outliers.extend(list(feature_outliers.index.values))

    # Find the data points that where considered outliers for more than one feature
    multi_feature_outliers = list((Counter(outliers) - Counter(set(outliers))).keys())

    # Remove the outliers, if any were specified
    good_data = data.drop(data.index[multi_feature_outliers]).reset_index(drop = True)
    return good_data

###################################################################################################################

