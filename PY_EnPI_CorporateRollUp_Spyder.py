# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:53:35 2020

@author: 7rp
"""

from matplotlib import pyplot as plt
from IPython import get_ipython
get_ipython().magic('reset -sf')

plt.close('all')

from PY_EnPI_Clone import EnPI_RollUp_GetFiles, EnPI_RollUp_Calculation


##### START OF PROGRAM #####

Select_Mode = 'List'
Save_Results = 'False'

company_name = 'Acme'
folder_loc = []
file_names = ['Acme_Hastings_NonRegression','Acme_Mclean','Acme_Minneapolis','Acme_Rochester']


# Get the file names
facility_results = EnPI_RollUp_GetFiles(Select_Mode, folder_loc, file_names)

# Get roll up results
EnPI_RollUpResults = EnPI_RollUp_Calculation(facility_results)

# Save results to json
if Save_Results == True:
    EnPI_RollUpResults.to_json('EnPI_CorporateRollUp_' + company_name + '.json')