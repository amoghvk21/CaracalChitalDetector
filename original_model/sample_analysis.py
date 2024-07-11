import matplotlib.pyplot as plt
import numpy as np
import pickle
import librosa
import random
import pytz
from datetime import datetime
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


WORKING_DIR = 'C:/Users/Amogh/OneDrive - University of Cambridge/Programming-New/CaracalChitalDetector/'


# Calculated in remove_noise.py
MEAN_C_CALL_FREQ_HIGH = 1822.8
MEAN_C_CALL_FREQ_LOW = 765.7
MEAN_DELTA_TIME = 1.534

def get_5_random(d):
        # Create list of all
        all_items = []
        for (utctimestamp, deviceno), times in d.items():
            for time in times:
                all_items.append((utctimestamp, deviceno, time))

        # Get 5 random
        random.shuffle(all_items)
        all_items = all_items[:5]
        
        return all_items


def get_items():
    # Get 5 random TPs/FPs for analysis
    with open(WORKING_DIR + 'data/py_obj/TPs_templating.pkl', 'rb') as f:
        TPs_templating = pickle.load(f)

    with open(WORKING_DIR + 'data/py_obj/FPs_templating_new.pkl', 'rb') as f:
        FPs_templating = pickle.load(f)

    print('TP:')
    TPs = get_5_random(TPs_templating)
    #TPs = [(1711540800, 213, 1185.088), (1711540800, 213, 1461.056), (1711540800, 204, 1180.512), (1711540800, 204, 1143.136), (1711458000, 217, 3140.576)]
    print()
    print('FPs:')
    FPs = get_5_random(FPs_templating)

    with open(WORKING_DIR + "data/py_obj/TP_FP_5_sample.pkl", 'wb') as f:
        pickle.dump((TPs, FPs), f)


def retrieve_items():
    with open(WORKING_DIR + "data/py_obj/TP_FP_5_sample.pkl", 'rb') as f:
        return pickle.load(f)


def display_items(all_items):
    # Table headings
    print('i\tdevice no\tdate\t\t\t\ttime')

    for i, (utctimestamp, deviceno, time) in enumerate(all_items):
        # Get filepath and open wav file
        timezone = pytz.timezone('Asia/Kathmandu')
        dt = datetime.fromtimestamp(utctimestamp, tz=timezone)
        
        # Print into
        print(f'{i+1}\t{deviceno}\t\t{dt}\t{time}')
        
        # Only 5 items (Used when the sample was larger). As only 5 items anyway, code is redundant
        if i == 5:
            break
    print()
    

def get_time_bounds_TP(items):
    with open(WORKING_DIR + 'data/py_obj/selectiontabledata.pkl', 'rb') as f:
        data = pickle.load(f)

    # List of (utctimestamp, deviceno, LB, UB)
    new_items = []

    # Go through each item and find the bounding box it is inside
    for utctimestamp, deviceno, time in items:
        times = data[utctimestamp, deviceno]
        for LB, UB, species in times:
            # Check if inside the bounding box
            if LB < time and UB > time and species == 'C':
                break
        new_items.append((utctimestamp, deviceno, LB, UB))
    
    return new_items


def get_time_bounds_FP(items):
    new_items = []

    for utctimestamp, deviceno, time in items:
        new_items.append((utctimestamp, deviceno, round(time-MEAN_DELTA_TIME, 3), round(time+MEAN_DELTA_TIME, 3)))

    return new_items


def generate_plots(items, type):

    all_mfccs = []
    all_mean_mfccs = []
    
    for i, (utctimestamp, deviceno, LB_time, UB_time) in enumerate(items):
        # Get filepath and open wav file
        timezone = pytz.timezone('Asia/Kathmandu')
        dt = datetime.fromtimestamp(utctimestamp, tz=timezone)
        aud, sr = librosa.load(WORKING_DIR + f'data/Test set/1 hour files/CAR{deviceno}_{dt.year}{str(dt.month).rjust(2, '0')}{str(dt.day).rjust(2, '0')}${str(dt.hour).rjust(2, '0')}{str(dt.minute).rjust(2, '0')}00_{utctimestamp}.wav')

        # Convert from seconds to no of samples and round to nearest integer
        LB_sample = round(LB_time*sr)
        UB_sample = round(UB_time*sr)

        # Chop audio to include only these relevant samples of that audio
        chopped_aud = aud[LB_sample:UB_sample]

        # Calcuate MFCC for this chhopped audio clip between the relevant frequencies and append to a list
        mfcc = librosa.feature.mfcc(y=chopped_aud, sr=sr, fmin=MEAN_C_CALL_FREQ_LOW, fmax=MEAN_C_CALL_FREQ_HIGH, n_mfcc=12)
        all_mfccs.append(mfcc)

        # PLOT 1:
        # Heatmaps for each call's MFCCs
        # Create plot
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time', sr=sr)
        plt.colorbar()
        plt.title(f'MFCC Heatmap of Type: {type} | UTC: {utctimestamp} | deviceno: {deviceno} | Time: {LB_time} to {UB_time}')
        plt.ylabel("MFCC Coefficients")
        plt.savefig(WORKING_DIR + f'plots/heatmaps/{type}_{i}_heatmap_mfcc_{utctimestamp}_{deviceno}_{str(LB_time)}_{str(UB_time)}.jpeg', format='jpeg')
        
        # PLOT 2:
        # Bar chart of the mean mfcc feature for all time
        mean_mfcc = []
        for mfccs in mfcc:
            mean_mfcc.append(np.mean(mfccs))
        all_mean_mfccs.append(mean_mfcc)
        
        plt.figure(figsize=(10, 4))
        plt.bar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], mean_mfcc)
        plt.title(f'Barchart of mean MFCC of Type: {type} | UTC: {utctimestamp} | deviceno: {deviceno} | Time: {LB_time} to {UB_time}')
        plt.xlabel("MFCC Coefficients")
        plt.ylabel("Mean Magnitude of the Coefficient")
        plt.savefig(WORKING_DIR + f'plots/barcharts/{type}_{i}_barchart_mean_mfcc_{utctimestamp}_{deviceno}_{str(LB_time)}_{str(UB_time)}.jpeg', format='jpeg')

        print(f'{type}_{i}')

    # PLOT 3 and 4:
    # Multiple bar chart for mean of each coefficient for all 5 samples of TP and FP

    # Normalise all coefficients so that you can compare them with each other
    all_mean_mfccs = np.array(all_mean_mfccs)
    all_mean_mfccs = all_mean_mfccs.T
    new = np.array([x/(np.sum(np.abs(x))) for x in all_mean_mfccs])
    print(new[0])
    all_mean_mfccs = new.T

    plt.figure(figsize=(10, 4))
    r = np.arange(12) 
    width = 0.1
    for i in range(5):
        plt.bar(r+(width*i), all_mean_mfccs[i], width=width, label=i) 
    plt.xticks(r + 5*width/2, ['1','2','3','4', '5', '6', '7', '8', '9', '10', '11', '12']) 
    plt.legend()
    plt.xlabel("MFCC Coefficients")
    plt.ylabel("Mean Magnitude of the Coefficients")
    plt.title(f'Mean barcharts for all {type}s')
    plt.savefig(WORKING_DIR + f'plots/{type}_multiplebarchart_mfcc_mean.jpeg', format='jpeg')

    return all_mean_mfccs


def generate_plots_2(all_mean_mfccs_TP, all_mean_mfccs_FP):
    
    # PLOT 5:
    # Take mean of average mfcc value for TP and FP and look at each side by side
    all_mean_mfccs_TP = np.array([np.array(x) for x in all_mean_mfccs_TP])
    all_mean_mfccs_FP = np.array([np.array(x) for x in all_mean_mfccs_FP])

    all_mean_mfccs_TP = all_mean_mfccs_TP.T
    all_mean_mfccs_FP = all_mean_mfccs_FP.T
    
    y_TP = [np.mean(y) for y in all_mean_mfccs_TP]
    y_FP = [np.mean(y) for y in all_mean_mfccs_FP]

    y = [y_TP, y_FP]
    label = ['TP', 'FP']

    plt.figure(figsize=(10, 4))
    r = np.arange(12) 
    width = 0.1
    for i in range(2):
        plt.bar(r+(width*i), y[i], width=width, label=label[i]) 
    plt.xticks(r + 2*width/2, ['1','2','3','4', '5', '6', '7', '8', '9', '10', '11', '12']) 
    plt.legend()
    plt.xlabel("MFCC Coefficients")
    plt.ylabel("Mean Magnitude of the Coefficients")
    plt.title("Mean barchart for TPs and FPs")
    plt.savefig(WORKING_DIR + f'plots/multiplebarchart_mfcc_mean_2.jpeg', format='jpeg')


def mean_square_difference_mfcc(items, type):
    
    # Stores list of all coefficients. For each internal list, there is a list of values from all 5 samples for their coefficients
    # Shape (12, 5, length)
    all_mfcc_values = [[], [], [], [], [], [], [], [], [], [], [], []]
    
    # Stores same data as all_mfcc_values but in different structure. 
    # List of each file and within each file, a 2D list of (12, length)
    # (5, 12, length)
    all_files = []

    for i, (utctimestamp, deviceno, LB_time, UB_time) in enumerate(items):
        # Get filepath and open wav file
        timezone = pytz.timezone('Asia/Kathmandu')
        dt = datetime.fromtimestamp(utctimestamp, tz=timezone)
        aud, sr = librosa.load(WORKING_DIR + f'data/Test set/1 hour files/CAR{deviceno}_{dt.year}{str(dt.month).rjust(2, '0')}{str(dt.day).rjust(2, '0')}${str(dt.hour).rjust(2, '0')}{str(dt.minute).rjust(2, '0')}00_{utctimestamp}.wav')

        # Convert from seconds to no of samples and round to nearest integer
        LB_sample = round(LB_time*sr)
        UB_sample = round(UB_time*sr)

        # Chop audio to include only these relevant samples of that audio
        chopped_aud = aud[LB_sample:UB_sample]

        # Calcuate MFCC for this chopped audio clip between the relevant frequencies and append to a list
        # Shape (12, no of samples)
        mfcc = librosa.feature.mfcc(y=chopped_aud, sr=sr, fmin=MEAN_C_CALL_FREQ_LOW, fmax=MEAN_C_CALL_FREQ_HIGH, n_mfcc=12)
        
        # Go through each coefficient and append it to the relevant list
        new_mfcc = []
        for j, values in enumerate(mfcc):
            # Normalise the values between 0 and 1
            values = values/np.sum(np.abs(values))
            all_mfcc_values[j].append(values)
            new_mfcc.append(values)
        new_mfcc = np.array(new_mfcc)

        all_files.append(new_mfcc)
    
    # Get the max value in coeff_1 - We center whole sample on this (i)
    # Get the min left and right distance - We trim all files to this so that same length
    coeff_1 = all_mfcc_values[0]
    indicies = []
    left = float('inf')
    right = float('inf')
    for file in coeff_1:
        indicies.append(np.argmax(file))
        left = min(left, indicies[-1])
        right = min(right, len(file)-indicies[-1])
    
    # Trim left and right of i and construct (12, 5, length) shape array
    new_all_mfcc_values = [[] for _ in range(12)]
    for coeff, values in enumerate(all_mfcc_values):
        for file, v in enumerate(values):
            i = indicies[file]
            new_all_mfcc_values[coeff].append(v[i-left:i+right])
    
    # Trim left and right of i and construct (5, 12, length) shape array
    new_all_files = [[] for _ in range(5)]
    for file, values in enumerate(all_files):
        for coeff, v in enumerate(values):
            i = indicies[file]
            new_all_files[file].append(v[i-left:i+right])

    # Plot each coeff
    for coeff, values in enumerate(new_all_mfcc_values):
        plt.figure(figsize=(10, 4))
        for file, v in enumerate(values):
            plt.plot(range(len(v)), v, label=str(file+1))
        plt.title(f'{type} {coeff+1} timestep against relative coefficient value for {len(new_all_files)} samples')
        plt.xlabel('Timestep')
        plt.ylabel('Relative coefficient value')
        plt.legend()
        plt.savefig(WORKING_DIR + f'plots/coeff_linegraph/{type}_{coeff+1}_linegraph.jpeg', format='jpeg')
        print(f'{type}_{coeff+1}')

    # Computing distance statistic
    print(f'Mean squared difference between all pairs of the {len(all_files)} heatmaps for {type} (After correcting for the peak):')
    new_all_files = np.array(new_all_files)

    # Go through each pair of heatmap
    for i, heatmap1 in enumerate(new_all_files):
        for heatmap2 in new_all_files[i+1:]:
            # Subtract, square, sum, divide by no of elements for mean
            total = np.shape(heatmap1)[0] * np.shape(heatmap2)[1]
            print((np.sum((heatmap1-heatmap2)**2))/total)


def mds(items_TP, items_FP):
    # Already saved therefore commented out
    '''
    # Stores list of all coefficients. For each internal list, there is a list of values from all 5 samples for their coefficients
    # Shape (12, 100, length)
    all_mfcc_values_TP = [[], [], [], [], [], [], [], [], [], [], [], []]
    
    # Stores same data as all_mfcc_values but in different structure. 
    # List of each file and within each file, a 2D list of (12, length)
    # (100, 12, length)
    all_files_TP = []

    for i, (utctimestamp, deviceno, LB_time, UB_time) in enumerate(items_TP):
        # Get filepath and open wav file
        timezone = pytz.timezone('Asia/Kathmandu')
        dt = datetime.fromtimestamp(utctimestamp, tz=timezone)
        aud, sr = librosa.load(WORKING_DIR + f'data/Test set/1 hour files/CAR{deviceno}_{dt.year}{str(dt.month).rjust(2, '0')}{str(dt.day).rjust(2, '0')}${str(dt.hour).rjust(2, '0')}{str(dt.minute).rjust(2, '0')}00_{utctimestamp}.wav')

        # Convert from seconds to no of samples and round to nearest integer
        LB_sample = round(LB_time*sr)
        UB_sample = round(UB_time*sr)

        # Chop audio to include only these relevant samples of that audio
        chopped_aud = aud[LB_sample:UB_sample]

        # Calcuate MFCC for this chopped audio clip between the relevant frequencies and append to a list
        # Shape (12, no of samples)
        mfcc = librosa.feature.mfcc(y=chopped_aud, sr=sr, fmin=MEAN_C_CALL_FREQ_LOW, fmax=MEAN_C_CALL_FREQ_HIGH, n_mfcc=12)
        
        # Go through each coefficient and append it to the relevant list
        new_mfcc = []
        for j, values in enumerate(mfcc):
            # Normalise the values between 0 and 1
            values = values/np.sum(np.abs(values))
            all_mfcc_values_TP[j].append(values)
            new_mfcc.append(values)
        new_mfcc = np.array(new_mfcc)

        all_files_TP.append(new_mfcc)
    
    # Same for FPs
    all_mfcc_values_FP = [[], [], [], [], [], [], [], [], [], [], [], []]
    all_files_FP = []
    for i, (utctimestamp, deviceno, LB_time, UB_time) in enumerate(items_FP):
        # Get filepath and open wav file
        timezone = pytz.timezone('Asia/Kathmandu')
        dt = datetime.fromtimestamp(utctimestamp, tz=timezone)
        aud, sr = librosa.load(WORKING_DIR + f'data/Test set/1 hour files/CAR{deviceno}_{dt.year}{str(dt.month).rjust(2, '0')}{str(dt.day).rjust(2, '0')}${str(dt.hour).rjust(2, '0')}{str(dt.minute).rjust(2, '0')}00_{utctimestamp}.wav')

        # Convert from seconds to no of samples and round to nearest integer
        LB_sample = round(LB_time*sr)
        UB_sample = round(UB_time*sr)

        # Chop audio to include only these relevant samples of that audio
        chopped_aud = aud[LB_sample:UB_sample]

        # Calcuate MFCC for this chopped audio clip between the relevant frequencies and append to a list
        # Shape (12, no of samples)
        mfcc = librosa.feature.mfcc(y=chopped_aud, sr=sr, fmin=MEAN_C_CALL_FREQ_LOW, fmax=MEAN_C_CALL_FREQ_HIGH, n_mfcc=12)
        
        # Go through each coefficient and append it to the relevant list
        new_mfcc = []
        for j, values in enumerate(mfcc):
            # Normalise the values between 0 and 1
            values = values/np.sum(np.abs(values))
            all_mfcc_values_FP[j].append(values)
            new_mfcc.append(values)
        new_mfcc = np.array(new_mfcc)

        all_files_FP.append(new_mfcc)

    with open(WORKING_DIR + f'data/py_obj/all_files_TP_FP.pkl', 'wb') as f:
        pickle.dump((all_files_TP, all_files_FP), f)

    with open(WORKING_DIR + f'data/py_obj/all_mfcc_values_TP_FP.pkl', 'wb') as f:
        pickle.dump((all_mfcc_values_TP, all_mfcc_values_FP), f)
    '''
        
    with open(WORKING_DIR + f'data/py_obj/all_files_TP_FP.pkl', 'rb') as f:
        all_files_TP, all_files_FP = pickle.load(f)
    
    with open(WORKING_DIR + f'data/py_obj/all_mfcc_values_TP_FP.pkl', 'rb') as f:
        all_mfcc_values_TP, all_mfcc_values_FP = pickle.load(f)

    # Get the max value in coeff_1 - We center whole sample on this (i)
    # Get the min left and right distance - We trim all files to this so that same length
    coeff_1 = all_mfcc_values_TP[0]
    indicies_TP = []
    left = float('inf')
    right = float('inf')
    for file in coeff_1:
        indicies_TP.append(np.argmax(file))
        left = min(left, indicies_TP[-1])
        right = min(right, len(file)-indicies_TP[-1])

    # Get the max value in coeff_1 - We center whole sample on this (i)
    # Get the min left and right distance - We trim all files to this so that same length
    coeff_1 = all_mfcc_values_TP[0]
    indicies_FP = []
    left = float('inf')
    right = float('inf')
    for file in coeff_1:
        indicies_FP.append(np.argmax(file))
        left = min(left, indicies_FP[-1])
        right = min(right, len(file)-indicies_FP[-1])
    
    # Trim left and right of i and construct (5, 12, length) shape array
    # Only store info about the first coefficient
    new_all_files_TP = [[] for _ in range(100)]
    for file, values in enumerate(all_files_TP):
        for coeff, v in enumerate(values):
            if coeff != 0:
                continue
            else:
                i = indicies_TP[file]
                new_all_files_TP[file].append(v[i-left:i+right])
    
    # Trim left and right of i and construct (5, 12, length) shape array
    # Only store info about the first coefficient
    new_all_files_FP = [[] for _ in range(100)]
    for file, values in enumerate(all_files_FP):
        for coeff, v in enumerate(values):
            if coeff != 0:
                continue
            else:
                i = indicies_FP[file]
                new_all_files_FP[file].append(v[i-left:i+right])

    # Remove any elements inside "new_all_files_FP" that have incorrect shape (due to trying to center at max)
    count = 0
    temp = []
    for i, x in enumerate(new_all_files_FP):
        if len(x[0]) == (left + right):
            temp.append(x)
        else:
            count += 1
    new_all_files_FP = temp
    del temp
    
    # Count this and also remove same amount of TPs
    new_all_files_TP = new_all_files_TP[:-1]

    # Go through all pairs of heatmaps and create a distance matrix (mat)
    s = len(new_all_files_TP) + len(new_all_files_FP)
    mat = np.zeros(shape=(s, s))
    combined = new_all_files_TP + new_all_files_FP
    combined = np.array(combined)
    for i, heatmap1 in enumerate(combined):
        for j, heatmap2 in enumerate(combined):
            # Subtract, square, sum, divide by no of elements for mean
            total = np.shape(heatmap1)[0] * np.shape(heatmap1)[1]
            mat[i][j] = (np.sum((heatmap1 - heatmap2) ** 2)) / total

    # Calculate the MDS coordinates
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_coordinates = mds.fit_transform(mat)

    # Plot the MDS
    plt.figure(figsize=(8, 6))
    for i, coord in enumerate(mds_coordinates):
        if i < 100:
            color = 'blue'
            label = 'TP'
        else:
            color = 'red'
            label = 'FP'
        plt.scatter(coord[0], coord[1], color=color, label=label)
    for i, txt in enumerate(range(len(combined))):
        plt.annotate(txt, (mds_coordinates[i, 0], mds_coordinates[i, 1]))
    plt.title('MDS Projection')
    plt.xlabel('MDS Component 1')
    plt.ylabel('MDS Component 2')
    plt.legend()
    plt.show()

    # Instead calculate the average difference between TP pairs and TP-FP pairs
    # Expect to see TPs difference be smaller (as more similar to eachother) than TP-FP pairs
    new_all_files_TP = np.array(new_all_files_TP)
    new_all_files_FP = np.array(new_all_files_FP)

    # TP pairs
    ans1 = []
    for i, heatmap1 in enumerate(new_all_files_TP):
        for heatmap2 in new_all_files_TP[i+1:]:
            # Subtract, square, sum, divide by no of elements for mean
            total = np.shape(heatmap1)[0] * np.shape(heatmap1)[1]
            ans1.append((np.sum((heatmap1 - heatmap2) ** 2)) / total)

    # Pairwise opposites
    ans2 = []
    for i, heatmap1 in enumerate(new_all_files_TP):
        for j, heatmap2 in enumerate(new_all_files_FP):
            # Subtract, square, sum, divide by no of elements for mean
            total = np.shape(heatmap1)[0] * np.shape(heatmap1)[1]
            ans2.append((np.sum((heatmap1 - heatmap2) ** 2)) / total)
    
    print()
    print(f'Mean squared difference between TP pairs: {np.mean(ans1)}')
    print(f'Mean squared difference between TP-FP pairs: {np.mean(ans2)}')
    print(f'Score2/Score1: {np.mean(ans2)/np.mean(ans1)}')
    print()

    # Now iterating through different dimensions of MDS and performing K means on each and calculating silhouette score
    results = [float('-inf')]
    for dim in range(1, 51):
        # Calculate the MDS coordinates
        mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=42)
        mds_coordinates = mds.fit_transform(mat)
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(mds_coordinates)
        labels = kmeans.predict(mds_coordinates)
        silhouette_avg = silhouette_score(mds_coordinates, labels)
        #print(f"{dim}: Silhouette Score: {silhouette_avg}")
        results.append(silhouette_avg)
    best_dim = np.argmax(results)
    print(f'Best no of dimensions for MDS between 1-50 is: {best_dim}')
    print(f'Prev silhouette value with 2 dimensions is: {results[2]}')
    print(f'New best silhouette value with {best_dim} dimensions is: {results[best_dim]}')
    
    def most_common(l):
        counts = np.bincount(l)
        return np.argmax(counts)
    
    results = []
    for dim in range(1, 51):
        mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=42)
        mds_coordinates = mds.fit_transform(mat)
        kmeans = KMeans(n_clusters=2, random_state=0)
        kmeans.fit(mds_coordinates)
        
        labels = kmeans.predict(mds_coordinates)
        labelsTP = labels[:100-count]
        labelsFP = labels[100-count:]
        
        TP_class = most_common(labelsTP)
        if TP_class == 1:
            FP_class = 0
        else:
            FP_class = 1
        
        TP_acc = np.count_nonzero(labelsTP == TP_class)/len(labelsTP)
        FP_acc = np.count_nonzero(labelsTP == FP_class)/len(labelsFP)
        
        avg = 2 / ((1 / TP_acc) + (1 / FP_acc))
        results.append(avg)
    best_dim = np.argmax(results)

    print()
    print(f'Best no of dimensions for MDS between 1-50 is: {best_dim}')
    print(f'Prev avg value with 2 dimensions is: {results[2]}')
    print(f'New best avg value with {best_dim} dimensions is: {results[best_dim]}')

    # Using optimal value of dimensions, see how well clustered the points are
    mds = MDS(n_components=best_dim, dissimilarity='precomputed', random_state=42)
    mds_coordinates = mds.fit_transform(mat)
    kmeans.fit(mds_coordinates)
    labels = kmeans.predict(mds_coordinates)
    labelsTP = labels[:100-count]
    labelsFP = labels[100-count:]
    print()
    print(f'No of class 0 in TPs: {np.count_nonzero(labelsTP == 0)}')
    print(f'No of class 1 in TPs: {np.count_nonzero(labelsTP == 1)}')
    print(f'No of class 0 in FPs: {np.count_nonzero(labelsFP == 0)}')
    print(f'No of class 1 in FPs: {np.count_nonzero(labelsFP == 1)}')


def get_100_random(d):
        # Create list of all
        all_items = []
        for (utctimestamp, deviceno), times in d.items():
            for time in times:
                all_items.append((utctimestamp, deviceno, time))

        # Get 5 random
        random.shuffle(all_items)
        all_items = all_items[:100]
        
        return all_items


def get_items_100():
    # Get 5 random TPs/FPs for analysis
    with open(WORKING_DIR + 'data/py_obj/TPs_templating.pkl', 'rb') as f:
        TPs_templating = pickle.load(f)

    with open(WORKING_DIR + 'data/py_obj/FPs_templating_new.pkl', 'rb') as f:
        FPs_templating = pickle.load(f)

    print('TPS:')
    TPs = get_100_random(TPs_templating)
    print()
    print('FPs:')
    FPs = get_100_random(FPs_templating)

    with open(WORKING_DIR + "data/py_obj/TP_FP_100_sample.pkl", 'wb') as f:
        pickle.dump((TPs, FPs), f)


def retrieve_items_100():
    with open(WORKING_DIR + "data/py_obj/TP_FP_100_sample.pkl", 'rb') as f:
        return pickle.load(f)


def plot_histogram(itemsTP, itemsFP):

    # Already saved therefore commented out
    '''
    # Stores list of all coefficients. For each internal list, there is a list of values from all 5 samples for their coefficients
    # Shape (12, 100, length)
    all_mfcc_values_TP = [[], [], [], [], [], [], [], [], [], [], [], []]
    
    # Stores same data as all_mfcc_values but in different structure. 
    # List of each file and within each file, a 2D list of (12, length)
    # (100, 12, length)
    all_files_TP = []

    for i, (utctimestamp, deviceno, LB_time, UB_time) in enumerate(items_TP):
        # Get filepath and open wav file
        timezone = pytz.timezone('Asia/Kathmandu')
        dt = datetime.fromtimestamp(utctimestamp, tz=timezone)
        aud, sr = librosa.load(WORKING_DIR + f'data/Test set/1 hour files/CAR{deviceno}_{dt.year}{str(dt.month).rjust(2, '0')}{str(dt.day).rjust(2, '0')}${str(dt.hour).rjust(2, '0')}{str(dt.minute).rjust(2, '0')}00_{utctimestamp}.wav')

        # Convert from seconds to no of samples and round to nearest integer
        LB_sample = round(LB_time*sr)
        UB_sample = round(UB_time*sr)

        # Chop audio to include only these relevant samples of that audio
        chopped_aud = aud[LB_sample:UB_sample]

        # Calcuate MFCC for this chopped audio clip between the relevant frequencies and append to a list
        # Shape (12, no of samples)
        mfcc = librosa.feature.mfcc(y=chopped_aud, sr=sr, fmin=MEAN_C_CALL_FREQ_LOW, fmax=MEAN_C_CALL_FREQ_HIGH, n_mfcc=12)
        
        # Go through each coefficient and append it to the relevant list
        new_mfcc = []
        for j, values in enumerate(mfcc):
            # Normalise the values between 0 and 1
            values = values/np.sum(np.abs(values))
            all_mfcc_values_TP[j].append(values)
            new_mfcc.append(values)
        new_mfcc = np.array(new_mfcc)

        all_files_TP.append(new_mfcc)
    
    # Same for FPs
    all_mfcc_values_FP = [[], [], [], [], [], [], [], [], [], [], [], []]
    all_files_FP = []
    for i, (utctimestamp, deviceno, LB_time, UB_time) in enumerate(items_FP):
        # Get filepath and open wav file
        timezone = pytz.timezone('Asia/Kathmandu')
        dt = datetime.fromtimestamp(utctimestamp, tz=timezone)
        aud, sr = librosa.load(WORKING_DIR + f'data/Test set/1 hour files/CAR{deviceno}_{dt.year}{str(dt.month).rjust(2, '0')}{str(dt.day).rjust(2, '0')}${str(dt.hour).rjust(2, '0')}{str(dt.minute).rjust(2, '0')}00_{utctimestamp}.wav')

        # Convert from seconds to no of samples and round to nearest integer
        LB_sample = round(LB_time*sr)
        UB_sample = round(UB_time*sr)

        # Chop audio to include only these relevant samples of that audio
        chopped_aud = aud[LB_sample:UB_sample]

        # Calcuate MFCC for this chopped audio clip between the relevant frequencies and append to a list
        # Shape (12, no of samples)
        mfcc = librosa.feature.mfcc(y=chopped_aud, sr=sr, fmin=MEAN_C_CALL_FREQ_LOW, fmax=MEAN_C_CALL_FREQ_HIGH, n_mfcc=12)
        
        # Go through each coefficient and append it to the relevant list
        new_mfcc = []
        for j, values in enumerate(mfcc):
            # Normalise the values between 0 and 1
            values = values/np.sum(np.abs(values))
            all_mfcc_values_FP[j].append(values)
            new_mfcc.append(values)
        new_mfcc = np.array(new_mfcc)

        all_files_FP.append(new_mfcc)

    with open(WORKING_DIR + f'data/py_obj/all_files_TP_FP.pkl', 'wb') as f:
        pickle.dump((all_files_TP, all_files_FP), f)

    with open(WORKING_DIR + f'data/py_obj/all_mfcc_values_TP_FP.pkl', 'wb') as f:
        pickle.dump((all_mfcc_values_TP, all_mfcc_values_FP), f)
    '''

    # Shape: (100, 12, length) 
    with open(WORKING_DIR + f'data/py_obj/all_files_TP_FP.pkl', 'rb') as f:
        all_files_TP, all_files_FP = pickle.load(f)
    
    # Shape: (12, 100, length) 
    with open(WORKING_DIR + f'data/py_obj/all_mfcc_values_TP_FP.pkl', 'rb') as f:
        all_mfcc_values_TP, all_mfcc_values_FP = pickle.load(f)

    # Get the max value in coeff_1 - We center whole sample on this (i)
    # Get the min left and right distance - We trim all files to this so that same length
    coeff_1 = all_mfcc_values_TP[0]
    indicies_TP = []
    left = float('inf')
    right = float('inf')
    for file in coeff_1:
        indicies_TP.append(np.argmax(file))
        left = min(left, indicies_TP[-1])
        right = min(right, len(file)-indicies_TP[-1])

    # Get the max value in coeff_1 - We center whole sample on this (i)
    # Get the min left and right distance - We trim all files to this so that same length
    coeff_1 = all_mfcc_values_TP[0]
    indicies_FP = []
    left = float('inf')
    right = float('inf')
    for file in coeff_1:
        indicies_FP.append(np.argmax(file))
        left = min(left, indicies_FP[-1])
        right = min(right, len(file)-indicies_FP[-1])
    
    # Trim left and right of i and construct (5, 12, length) shape array
    # Only store info about the first coefficient
    new_all_files_TP = [[] for _ in range(100)]
    for file, values in enumerate(all_files_TP):
        for coeff, v in enumerate(values):
            if coeff != 0:
                continue
            else:
                i = indicies_TP[file]
                new_all_files_TP[file].append(v[i-left:i+right])
    
    # Trim left and right of i and construct (5, 12, length) shape array
    # Only store info about the first coefficient
    new_all_files_FP = [[] for _ in range(100)]
    for file, values in enumerate(all_files_FP):
        for coeff, v in enumerate(values):
            if coeff != 0:
                continue
            else:
                i = indicies_FP[file]
                new_all_files_FP[file].append(v[i-left:i+right])

    # Remove any elements inside "new_all_files_FP" that have incorrect shape (due to trying to center at max)
    count = 0
    temp = []
    for i, x in enumerate(new_all_files_FP):
        if len(x[0]) == (left + right):
            temp.append(x)
        else:
            count += 1
    new_all_files_FP = temp
    del temp
    
    # Count this and also remove same amount of TPs
    new_all_files_TP = new_all_files_TP[:-1]

    # Calculate the average difference between TP pairs and TP-FP pairs
    # Expect to see TPs difference be smaller (as more similar to eachother) than TP-FP pairs
    new_all_files_TP = np.array(new_all_files_TP)
    new_all_files_FP = np.array(new_all_files_FP)

    # TP pairs
    ans1 = []
    for i, heatmap1 in enumerate(new_all_files_TP):
        for heatmap2 in new_all_files_TP[i+1:]:
            # Subtract, square, sum, divide by no of elements for mean
            total = np.shape(heatmap1)[0] * np.shape(heatmap1)[1]
            ans1.append((np.sum((heatmap1 - heatmap2) ** 2)) / total)

    # Pairwise opposites
    ans2 = []
    for i, heatmap1 in enumerate(new_all_files_TP):
        for j, heatmap2 in enumerate(new_all_files_FP):
            # Subtract, square, sum, divide by no of elements for mean
            total = np.shape(heatmap1)[0] * np.shape(heatmap1)[1]
            ans2.append((np.sum((heatmap1 - heatmap2) ** 2)) / total)
    
    print()
    print(f'Mean squared difference between TP pairs: {np.mean(ans1)}')
    print(f'Mean squared difference between TP-FP pairs: {np.mean(ans2)}')
    print(f'Score2/Score1: {np.mean(ans2)/np.mean(ans1)}')
    print()
    
    ans1_new = []
    for x in ans1:
        y = np.log(x)
        if y == float('-inf'):
            continue
        else:
            ans1_new.append(y)

    ans2_new = []
    for x in ans2:
        y = np.log(x) 
        if y == float('-inf'):
            continue
        else:
            ans2_new.append(y)

    # Calculate histogram data
    counts1, bin_edges1 = np.histogram(ans1_new, bins=30, density=True)
    counts2, bin_edges2 = np.histogram(ans2_new, bins=30, density=True)

    # Calculate the bin centers
    bin_centers1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
    bin_centers2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2 

    # Plot histogram as a line plot
    plt.plot(-1*np.log(bin_centers1*-1), counts1, color='blue', label='TP-TP pairs difference')
    plt.plot(-1*np.log(bin_centers2*-1), counts2, color='red', label='TP-FP pairs difference')
    #plt.axvline(x=np.mean(ans1), color='blue', linestyle='--', linewidth=2, label='TP-TP pairs average difference')
    #plt.axvline(x=np.mean(ans2), color='red', linestyle='--', linewidth=2, label='TP-FP pairs average difference')
    plt.xlabel('Squared difference between heatmaps')
    plt.ylabel('Density')
    plt.title('Histogram of difference between TP-TP and TP-FP pairs')
    plt.legend()
    plt.show()


def main():

    # Plotting
    '''
    #get_items()  # Comment out once you have got and saved your 10 items for analysis

    TPs, FPs = retrieve_items()

    print("TPs:")
    display_items(TPs)
    print("FPs:")
    display_items(FPs)

    TP_times = get_time_bounds_TP(TPs)
    FP_times = get_time_bounds_FP(FPs)
    
    meanTP = generate_plots(TP_times, 'TP')
    meanFP = generate_plots(FP_times, 'FP')

    generate_plots_2(meanTP, meanFP)
    '''

    ##############################################

    # Mean squared difference for the 10 pairs of TP and 10 pairs of FP...
    # ...and line graphs for each coefficient
    '''
    #get_items()  # Comment out once you have got and saved your 10 items for analysis

    TPs, FPs = retrieve_items()

    print("TPs:")
    display_items(TPs)
    print("FPs:")
    display_items(FPs)

    TP_times = get_time_bounds_TP(TPs)
    FP_times = get_time_bounds_FP(FPs)

    mean_square_difference_mfcc(TP_times,  'TP')
    print()
    mean_square_difference_mfcc(FP_times, 'FP')
    '''

    ##############################################

    # MDS plot for 100 samples
    # and for each iteration of MDS, perform DFA
    '''
    #get_items_100()  # Comment out once you have got and saved your 100 items for analysis

    TPs, FPs = retrieve_items_100()

    print('displaying first few items...')
    print("TPs:")
    display_items(TPs)
    print("FPs:")
    display_items(FPs)

    TP_times = get_time_bounds_TP(TPs)
    FP_times = get_time_bounds_FP(FPs)

    print('running mds()')
    mds(TP_times, FP_times)
    '''
    
    ##############################################
    
    # Plotting histogram of mean squared distance between TP-TP pairs and TP-FP pairs to find separation
    #'''
    #get_items_100()  # Comment out once you have got and saved your 100 items for analysis
    TPs, FPs = retrieve_items_100()
    plot_histogram(TPs, FPs)
    #'''


if __name__ == '__main__':
    main()