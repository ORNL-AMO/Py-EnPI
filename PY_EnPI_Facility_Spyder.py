# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 23:02:20 2020

@author: 7rp
"""

from matplotlib import pyplot as plt
from IPython import get_ipython
get_ipython().magic('reset -sf')

plt.close('all')

from PY_EnPI_Clone import EnPI_Load_Data, EnPI_PythonEdition


# Specify the path and file name for excel file containing facility data
folder_path = ''
file_name = 'Acme_Hastings.xlsx'
sheet_name = 'Hastings'

# Enter the columns in the excel file with the data to be analyzed
Data_Columns = 'A,C:E'

# Give the name of the variables used for unit of output
Production_Header = "Production (tons)"

# Give the number of utilities and relevant variables
n_utilities = 2
n_relevant_variables = 1

# Specify the range for Better Plants Reporting
Reporting_Range_Start = '2007-01-01'
Reporting_Range_End = '2010-01-01'

# Choose Method (Regression or Energy Intensity)
Method = "Energy Intensity"

# Show Regression Models (if True, shows all valid models but does not do EnPI analysis)
Show_Models = False

# Index of Chosen Models (-1 for best adjusted R2, [] for model information)
Chosen_Models = []

# Specify range for models years to be tested (leave as '' if same as reporting range)
Model_Range_Start = ''
Model_Range_End = ''

# Save the Results?
Save_Results = False



# Get data from Excel File
Facility_Data = EnPI_Load_Data(folder_path + file_name, sheet_name, Data_Columns, n_utilities, n_relevant_variables)

# Get the EnPI Results
EnPI_Results = EnPI_PythonEdition(Facility_Data, n_utilities, n_relevant_variables, Reporting_Range_Start, Reporting_Range_End, Model_Range_Start, Model_Range_End, Chosen_Models, Method, Show_Models, Production_Header)

