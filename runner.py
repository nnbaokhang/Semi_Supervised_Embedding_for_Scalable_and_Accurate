import numpy as np
import subprocess
import os
import glob
#NUM_TRIES = 5
#NUM_EPOCHS = 200
NUM_TRIES = 1
NUM_EPOCHS = 20
#DATASETS=["UWaveGestureLibraryAll","FacesUCR","ECG5000"]
#DATASETS=["Datasets/CDOT/Time_Series_For_Clustering_El_Paso_with_Weather_data.csv"]
DATASETS=["CDOT"]
exmpl_range = np.arange(4, 29, step=8) # What is the assumption here?


#DATASETS = [os.path.split(x)[-1] for x in glob.glob("D://Datasets/UCRArchive_2018/**")]
for DATASET in DATASETS:
    for x in exmpl_range:
        print("Number of examples: %d" % x)
        for i in range(NUM_TRIES):
            seed = np.random.randint(0, np.iinfo(np.int32).max)
            '''
            print("Using Seed %d" % seed)
            # Test Silh + AE
            print("Starting trial %d of %d with Silh=%s,Proto=%s,AE=%s" % (i+1,NUM_TRIES,True,False,True))
            #subprocess.run(["python","semi_supervised.py","--dataset=%s"%DATASET,"--number_examples=%d" % x,"--number_epochs=%d" % NUM_EPOCHS,"--silh","--ae","--seed=%d"%seed],check=True)
            # Test Silh on its own
            print("Starting trial %d of %d with Silh=%s,Proto=%s,AE=%s" % (i+1,NUM_TRIES,True,False,False))
            subprocess.run(["python","semi_supervised.py","--dataset=%s"%DATASET,"--number_examples=%d" % x,"--number_epochs=%d" % NUM_EPOCHS,"--silh","--seed=%d"%seed],check=True)
            # Test Proto + AE
            print("Starting trial %d of %d with Silh=%s,Proto=%s,AE=%s" % (i+1,NUM_TRIES,False,True,True))
            subprocess.run(["python","semi_supervised.py","--dataset=%s"%DATASET,"--number_examples=%d" % x,"--number_epochs=%d" % NUM_EPOCHS,"--ae","--proto","--seed=%d"%seed],check=True)
            # Test Proto
            print("Starting trial %d of %d with Silh=%s,Proto=%s,AE=%s" % (i+1,NUM_TRIES,False,True,False))
            subprocess.run(["python","semi_supervised.py","--dataset=%s"%DATASET,"--number_examples=%d" % x,"--number_epochs=%d" % NUM_EPOCHS,"--proto","--seed=%d"%seed],check=True)
            # Test AE
            print("Starting trial %d of %d with Silh=%s,Proto=%s,AE=%s" % (i+1,NUM_TRIES,False,False,True))
            subprocess.run(["python","semi_supervised.py","--dataset=%s"%DATASET,"--number_examples=%d" % x,"--number_epochs=%d" % NUM_EPOCHS,"--ae","--seed=%d"%seed],check=True)
            #Todo debug DB loss issue with tensorflow expands dims
            # Test DB + AE
            '''

            #print("Starting trial %d of %d with Silh=%s,Proto=%s,AE=%s,DB=%s" %(i+1,NUM_TRIES,False,False,True,True))
            #subprocess.run(["python","semi_supervised.py","--dataset=%s"%DATASET,"--number_examples=%d" % x,"--number_epochs=%d" % NUM_EPOCHS,"--ae", "--db", "--seed=%d"%seed],check=True)
            # Test DB
            print("Starting trial %d of %d with Silh=%s,Proto=%s,AE=%s,DB=%s" %(i+1,NUM_TRIES,False,False,False,True))
            subprocess.run(["python","semi_supervised.py","--dataset=%s"%DATASET,"--number_examples=%d" % x,"--number_epochs=%d" % NUM_EPOCHS,"--db", "--seed=%d"%seed],check=True)
