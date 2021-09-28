import matplotlib.pyplot as plt
import time
from pyts.image import RecurrencePlot
import pandas as pd
import multiprocessing

exoTrain = pd.read_csv(r'C:\Semester-1\ACML\Project-ExoPlanet-Search\Latest\archive\exoTrain.csv')
exoTest = pd.read_csv(r'C:\Semester-1\ACML\Project-ExoPlanet-Search\Latest\archive\exoTest.csv')




def extract_recu(X,y,i):
    
    '''
       Extracts Recurrence plots.
    '''
    
    inpt = X
    otpt = y
    k = i
    rp = RecurrencePlot()
    X_rp = rp.fit_transform(inpt,otpt)

    # Show the results for the first time series
    plt.figure(figsize=(5, 5))
    plt.imshow(X_rp[0])
    #plt.title('Recurrence Plot', fontsize=16)
    #plt.tight_layout()
    plt.axis('off')

    
    return plt.savefig(r'C:\Semester-1\ACML\Project-ExoPlanet-Search\Test\observation'+ str(k) + '.png',bbox_inches='tight',pad_inches = 0) #Change your address here


#extract_recu(exoTrain)

#start = time.time()
def multiprocessing_func(x):
    '''

        This function creates parallel processing to take cut time. In our case 
        we took 2 hours to creates plots where estimated time from single process
        was 18 - 20 hours.

    '''
    k = [0,190,380]    # Change for Train set
    z = [190,380,570] 
    #k = [2544,3393,4242]
    #z = [3393,4242,5087]
    for i in range(k[x],z[x]):
        X = exoTrain.loc[i:i, exoTrain.columns != 'LABEL']
        y = exoTrain.loc[i:i, exoTrain.columns == 'LABEL']
        
        #X = exoTest.loc[i:i, exoTest.columns != 'LABEL'] # Test set
        #y = exoTest.loc[i:i, exoTest.columns == 'LABEL'] # Test set
        extract_recu(X,y,i)


if __name__ == '__main__':
    starttime = time.time()
    processes = []
    for i in range(0,3):
        p = multiprocessing.Process(target=multiprocessing_func, args=(i,))
        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()
        
    print('That took {} seconds'.format(time.time() - starttime))

