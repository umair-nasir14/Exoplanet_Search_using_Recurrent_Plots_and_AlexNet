import pandas as pd
import matplotlib.pyplot as plt

'''
   Extract light curve time series plot 
'''

exoTrain = pd.read_csv(r'C:\Semester-1\ACML\Project-ExoPlanet-Search\Latest\archive\exoTrain.csv')
exoTest = pd.read_csv(r'C:\Semester-1\ACML\Project-ExoPlanet-Search\Latest\archive\exoTest.csv')


# =============================================================================
# FOR TRAINING SET
# for i in range(0,5088):
#     
#     plt.plot(exoTrain.iloc[i,exoTrain.columns != 'LABEL']    )
#     plt.axis('off')
#     plt.savefig(r'C:/Semester-1/ACML/Project-ExoPlanet-Search/Latest/graph_data/Train/graph_obs'+ str(i) + '.png',bbox_inches='tight',pad_inches = 0)
#     plt.show()
# =============================================================================
    
# =============================================================================
# FOR TEST SET
for i in range(0,570):
     
   plt.plot(exoTest.iloc[i,exoTest.columns != 'LABEL']    )
   plt.axis('off')
   plt.savefig(r'C:/Semester-1/ACML/Project-ExoPlanet-Search/Latest/graph_data/Data/Test/graph_obs'+ str(i) + '.png',bbox_inches='tight',pad_inches = 0)
   plt.show()
