# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:12:57 2020

@author: 7rp
"""

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import statsmodels.api as sm
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from itertools import combinations
from matplotlib import pyplot as plt
from dateutil.relativedelta import relativedelta
from matplotlib import patches
from statsmodels.stats.outliers_influence import variance_inflation_factor
from os import path
from tkinter import Tk, filedialog

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

pd.set_option('precision', 0)
pd.options.display.float_format = '{:,.2f}'.format

#%% Class Defitions

# Class exception to stop Jupyter from outputing entire exception trace
class StopExecution(Exception):
    def _render_traceback_(self):
        pass



#%% Loads data from excel file and checks for a couple of errors
def EnPI_Load_Data(flnm, sheet_name, ExcelColumns, n_utilities, n_vars):
    # Read data from file
    try:
        excel_file = pd.ExcelFile(flnm)
    except FileNotFoundError:
        print('******** ERROR ********')
        print('    FILE NOT FOUND!')
        print('***********************')
        print('\n')
        print('Solution:')
        print('Please check file name and path.')

        # sys.exit()
        raise StopExecution
    
    if sheet_name == '':
        sheet_name = flnm[5:-5]
    
    Plt_AllData = pd.read_excel(excel_file,sheet_name,usecols=ExcelColumns).dropna()
    
    if (len(Plt_AllData.columns)-1) != (n_utilities+n_vars):
        print('********* ERROR *********')
        print('WRONG NUMBER OF COLUMNS!')
        print('*************************')
        print('\n')
        print('Solution:')
        print('Please check number of utilities and relevent variables.')
        print('- OR -')
        print('Please check the specified range of excel columns.')
        

        sys.exit()
        # raise StopExecution
    
    return Plt_AllData

#%% Based on method, will direct program to the right subroutine
def EnPI_PythonEdition(Facility_Data, n_utilities, n_relevant_variables, Reporting_Range_Start, Reporting_Range_End, Model_Range_Start, Model_Range_End, Chosen_Models, Method, Show_Models, Production_Header):
    
    if Method == 'Regression':
        print('\n\nFinding Regression Models...\n')
        
        return EnPI_RegressionMethod(Facility_Data, n_utilities, n_relevant_variables, Reporting_Range_Start, Reporting_Range_End, Model_Range_Start, Model_Range_End, Chosen_Models, Show_Models)
    
    elif Method == 'Energy Intensity':
        print('\n\nFinding Energy Intensity...\n')
        
        return EnPI_EnergyIntensityMethod(Facility_Data, n_utilities, n_relevant_variables, Reporting_Range_Start, Reporting_Range_End, Production_Header)
    
    else:
        print('***** ERROR *****')
        print(  'UNKOWN METHOD!')
        print('*****************')
        print('\n')
        print('Solution:')
        print('Please check specified method is either "Regression" or "Energy Intensity".')

        raise StopExecution
    
    return 'Hi There!'

#%% Energy Intensity Method
def EnPI_EnergyIntensityMethod(Facility_Data, n_utilities, n_relevant_variables, Reporting_Range_Start, Reporting_Range_End, Production_Header):
    
    # Convert date strings to datetime format
    Reporting_Range_Start = dt.datetime.strptime(Reporting_Range_Start, '%Y-%m-%d')
    Reporting_Range_End= dt.datetime.strptime(Reporting_Range_End, '%Y-%m-%d')
    Reporting_yr_idx = (Reporting_Range_Start <= Facility_Data['Date']) & (Facility_Data['Date'] < Reporting_Range_End)
    
    # Get reporting range data
    Reporting_Dates = pd.to_datetime(Facility_Data['Date'][Reporting_yr_idx])
    Reporting_Energy = Facility_Data.iloc[:,1:n_utilities+1][Reporting_yr_idx]
    Reporting_Vars = Facility_Data.iloc[:,n_utilities+1:][Reporting_yr_idx]
    
    del Facility_Data
    
    # Make sure that specified production header is in the data
    try:
        Reporting_Vars = Reporting_Vars.loc[:,Production_Header]
    except KeyError:
        print('*********** ERROR ***********')
        print('PRODUCTION HEADER NOT FOUND!')
        print('*****************************')
        print('\n')
        print('Solution:')
        print('Please check that specified header for production is correct.')
        
        raise StopExecution
    
    # Get the number of years in the reporting range
    n_years = relativedelta(Reporting_Range_End, Reporting_Range_Start).years
    
    # Check to make sure that there is any data
    if Reporting_Vars.empty:
        print('*************** ERROR ***************')
        print('NO DATA IN SPECIFIED REPORTING RANGE!')
        print('*************************************')
        print('\n')
        print('Solution:')
        print('Please check that specified reporting range matches data in excel.')
        
        raise StopExecution
    
    # Make sure that there are complete years in the reporting range
    if (Reporting_Range_Start + relativedelta(years=n_years)) != Reporting_Range_End:
        print('************* ERROR ************')
        print('PARTIAL YEAR IN REPORTING RANGE!')
        print('********************************')
        print('\n')
        print('Solution:')
        print('Please check that specified reporting range has complete years.')
    
        raise StopExecution
        
    # Get the number of samples per year
    n_samples = len(Reporting_Vars) // n_years
    
    # Plot Reporting Data
    plt.figure()
    gca = plt.gca()
        
    # Plot each utility seperately
    for i in range(0,n_utilities):
        plt.plot(Reporting_Dates,Reporting_Energy.iloc[:,i],linestyle='-',marker='o', markersize=5)

    # Plot the total energy        
    plt.plot(Reporting_Dates,Reporting_Energy.sum(axis=1), linestyle=':', marker='o', markersize=5)
       
    # Adjust position of axis so legend can be placed outside and below
    box = gca.get_position()
    gca.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height*0.9])
    
    # Get legend labels for energy plot and add legend to plot   
    Energy_Legend = list(Reporting_Energy.columns)
    Energy_Legend.append('Total Energy')
    plt.legend(Energy_Legend,loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
    
    # Energy plot formatting
    plt.xticks(Reporting_Dates[0] + np.arange(n_years + 1)*relativedelta(months=12))
    plt.xlim([Reporting_Range_Start, Reporting_Range_End])
    plt.title('Energy Data')       
    plt.grid()
    
    
    
    # ***** CREATE ENPI RESULTS TABLE ***** #
    
    # Define rows for results table
    EnPI_Results_Indices = list('Actual ' + Reporting_Energy.columns)
    EnPI_Results_Indices.append('TOTAL ACTUAL ENERGY (MMBtu)')
    EnPI_Results_Indices.append('TOTAL PRODUCTION OUTPUT')
    EnPI_Results_Indices.append('PRODUCTION ENERGY INTENSITY (MMBtu/Unit Production)')
    EnPI_Results_Indices.append('Total Improvement in Energy Intensity (%)')
    EnPI_Results_Indices.append('Annual Improvement in Energy Intensity (%)')
    EnPI_Results_Indices += list(Reporting_Energy.columns + ' Annual Savings')
    EnPI_Results_Indices.append('New Energy Savings for Current Year (MMBtu/Year)')
    EnPI_Results_Indices.append('TOTAL ENERGY SAVINGS SINCE BASELINE YEAR (MMBtu/Year)')
    
    # Create EnPI Results like table
    EnPI_Results = pd.DataFrame(columns=range(1,n_years+1), index = EnPI_Results_Indices)
    
    
    
    # ***** ADD ENERGY INTENSITY ANALYSIS RESULTS TO TABLE ***** #
    
    # Total energy by year and utility
    EnPI_Results.iloc[0:n_utilities] = Reporting_Energy.groupby(Reporting_Vars.index // n_samples).sum().transpose().values
    
    # Total energy by year
    EnPI_Results.iloc[n_utilities] = EnPI_Results.iloc[0:n_utilities].sum()
    
    # Total production by year
    EnPI_Results.iloc[n_utilities+1] = Reporting_Vars.groupby(Reporting_Vars.index // n_samples).sum().transpose().values
    
    # Energy Intensity by year
    EnPI_Results.loc['PRODUCTION ENERGY INTENSITY (MMBtu/Unit Production)'] = np.divide(EnPI_Results.loc['TOTAL ACTUAL ENERGY (MMBtu)'], EnPI_Results.loc['TOTAL PRODUCTION OUTPUT'])
    
    # Set all baseline year savings to be 0
    EnPI_Results.iloc[-6:,0] = 0
    
    # Total Improvement in Energy Intensity
    EnPI_Results.loc['Total Improvement in Energy Intensity (%)'].iloc[1:] = (
        EnPI_Results.loc['PRODUCTION ENERGY INTENSITY (MMBtu/Unit Production)'].iloc[0] - 
        EnPI_Results.loc['PRODUCTION ENERGY INTENSITY (MMBtu/Unit Production)'].iloc[1:]) / (
        EnPI_Results.loc['PRODUCTION ENERGY INTENSITY (MMBtu/Unit Production)'].iloc[0]) * 100

    # Annual Improvement in Energy Intensity
    EnPI_Results.loc['Annual Improvement in Energy Intensity (%)'].iloc[1:] = np.diff(EnPI_Results.loc['Total Improvement in Energy Intensity (%)'])
    
    # Annual Energy Savings by year and utility
    EnPI_Results.iloc[-2-n_utilities:-n_utilities,1:] = -EnPI_Results.iloc[0:n_utilities,1:].subtract(EnPI_Results.iloc[0:n_utilities,0],axis=0).values
    
    # New current year energy savings and total energy savings
    EnPI_Results.loc['New Energy Savings for Current Year (MMBtu/Year)'].iloc[1:] = np.diff(-EnPI_Results.loc['TOTAL ACTUAL ENERGY (MMBtu)'].values)
    EnPI_Results.loc['TOTAL ENERGY SAVINGS SINCE BASELINE YEAR (MMBtu/Year)'].iloc[1:] = EnPI_Results.loc['TOTAL ACTUAL ENERGY (MMBtu)'].iloc[0] - EnPI_Results.loc['TOTAL ACTUAL ENERGY (MMBtu)'].iloc[1:]
    
    print('Done!')
    
    
    
    # ***** ENERGY INTENSITY IMPROVEMENT PLOT ***** #
    
    # Make energy intentisy plot
    plt.figure()
    gca = plt.gca()
    
    # Get energy intensity results
    EI = EnPI_Results.loc['PRODUCTION ENERGY INTENSITY (MMBtu/Unit Production)'].values
    
    # Bar plot of energy inentisy
    gca.bar(range(1,n_years+1), EI, 0.4)
    
    # Get value of goal energy intensity
    Goal_EI = 0.8*EI[0]
    # plt.plot([0.5, n_years+0.5], Goal_EI*np.ones(2), linestyle='--', color='k')
    
    # Adjust axis limits based on reporting years and range of EIs
    gca.set_ylim([Goal_EI // 1, -(np.max(EI) // -1)])
    gca.set_xlim([0.5, n_years+0.5])
    
    # Create a second y axis
    axs2 = gca.twinx()
    
    # Advance colormap to next color (this could break on a future release of MatPlotLib)
    axs2._get_lines.get_next_color()
    
    # Plot EI improvements
    axs2.plot(range(1,n_years+1), EnPI_Results.loc['Annual Improvement in Energy Intensity (%)'].values, marker='o', markersize=5, linewidth=2)
    axs2.plot(range(1,n_years+1), EnPI_Results.loc['Total Improvement in Energy Intensity (%)'].values, marker='x', markersize=7.5, linewidth=2, markeredgewidth=2)
    
    # Add legend to plot
    axs2.legend(['Annual Improvement in EI', 'Total Improvement in EI'],loc=2,fontsize=9)
    
    # Foramt the plot
    gca.set_xlabel('Reporting Year')
    gca.set_xticks(range(1,n_years + 1))
    gca.set_ylabel('Energy Intensity (MMBtu/Unit)')
    axs2.set_ylabel('Percent Improvement')
    
    # Make second y axis display percents
    axs2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    
    # Nudge axis to the left to better center everthing in figure window
    box = gca.get_position()
    gca.set_position([box.x0-0.015, box.y0, box.width, box.height])
    
    # Return the results
    return EnPI_Results

#%% EnPI Regression Method
def EnPI_RegressionMethod(Facility_Data, n_utilities, n_relevant_variables, Reporting_Range_Start, Reporting_Range_End, Model_Range_Start, Model_Range_End, Chosen_Models, Show_Models):
    
    # Convert date strings to datetime format
    Reporting_Range_Start = dt.datetime.strptime(Reporting_Range_Start, '%Y-%m-%d')
    Reporting_Range_End= dt.datetime.strptime(Reporting_Range_End, '%Y-%m-%d')
    Reporting_rng_idx = (Reporting_Range_Start <= Facility_Data['Date']) & (Facility_Data['Date'] < Reporting_Range_End)
    
    # Get reporting range data
    Reporting_Dates = pd.to_datetime(Facility_Data['Date'][Reporting_rng_idx])
    Reporting_Energy = Facility_Data.iloc[:,1:n_utilities+1][Reporting_rng_idx]
    Reporting_Vars = Facility_Data.iloc[:,n_utilities+1:][Reporting_rng_idx]
    
    # Check to make sure that there is any data
    if Reporting_Vars.empty:
        print('*************** ERROR ***************')
        print('NO DATA IN SPECIFIED REPORTING RANGE!')
        print('*************************************')
        print('\n')
        print('Solution:')
        print('Please check that specified reporting range matches data in excel.')
        
        raise StopExecution
    
    # Get the number of years in the reporting range
    n_years = relativedelta(Reporting_Range_End, Reporting_Range_Start).years
    
    
    # Make sure that there are complete years in the reporting range
    if (Reporting_Range_Start + relativedelta(years=n_years)) != Reporting_Range_End:
        print('************* ERROR ************')
        print('PARTIAL YEAR IN REPORTING RANGE!')
        print('********************************')
        print('\n')
        print('Solution:')
        print('Please check that specified reporting range has complete years.')
    
        raise StopExecution
    
    # Get possible model range data if different than the reporting range
    if Model_Range_Start != '':
        # Convert date strings to datetime format
        Model_Range_Start = dt.datetime.strptime(Model_Range_Start, '%Y-%m-%d')
        Model_Range_End= dt.datetime.strptime(Model_Range_End, '%Y-%m-%d')
        Model_rng_idx = (Model_Range_Start <= Facility_Data['Date']) & (Facility_Data['Date'] < Model_Range_End)
        
        # Get model range data
        Model_Dates = pd.to_datetime(Facility_Data['Date'][Model_rng_idx])
        Model_Energy = Facility_Data.iloc[:,1:n_utilities+1][Model_rng_idx]
        Model_Vars = Facility_Data.iloc[:,n_utilities+1:][Model_rng_idx]
        
        # Check to make sure that there is any data
        if Model_Vars.empty:
            print('************* ERROR *************')
            print('NO DATA IN SPECIFIED MODEL RANGE!')
            print('*********************************')
            print('\n')
            print('Solution:')
            print('Please check that specified reporting range matches data in excel.')
            
            raise StopExecution
            
        # Get the number of years in the reporting range
        n_model_years = relativedelta(Model_Range_End, Model_Range_Start).years
        
        # Make sure that there are complete years in the reporting range
        if (Model_Range_Start + relativedelta(years=n_model_years)) != Model_Range_End:
            print('*********** ERROR **********')
            print('PARTIAL YEAR IN MODEL RANGE!')
            print('****************************')
            print('\n')
            print('Solution:')
            print('Please check that specified reporting range has complete years.')
        
            raise StopExecution
        
    else:
        # Copy reporting range parameters
        Model_Range_Start = Reporting_Range_Start
        Model_Range_End = Reporting_Range_End
        Model_rng_idx = Reporting_rng_idx
        n_model_years = n_years
        
        # Get model range data
        Model_Dates = Reporting_Dates
        Model_Energy = Reporting_Energy
        Model_Vars = Reporting_Vars
        
    del Facility_Data

    # Get the number of data points in a year
    n_samples = len(Reporting_Vars) // n_years   
    
    # Create an empty list of model year indices
    Model_yr_idxs = n_model_years*[None]
    
    # Create a list of model year indices
    for i in range(0,n_model_years):
        Model_yr_idxs[i] = (Model_Range_Start + i*relativedelta(years=1) <= Model_Dates) & (Model_Dates < Model_Range_Start + (i+1)*relativedelta(years=1))
    
    
    
    # *****  FIND ALL VALID REGRESSION MODELS ***** #
    
    # Create a list of all valid models for each utility
    EnPI_Valid_Models = [None]*n_model_years
    
    # Create empty list to hold all of the model ranges
    Model_Year = [None]*n_model_years

    # Cycle through all of the possible model years to find valid models
    for i1 in range(0,len(Model_yr_idxs)):

        # Create an empty list for storing current model year regression models        
        EnPI_Models = list()
        
        # Define the current model year range
        Model_Start = Model_Dates.loc[Model_yr_idxs[i1]].iloc[0].strftime('%Y-%m-%d')
        Model_End = (Model_Dates.loc[Model_yr_idxs[i1]].iloc[0] + relativedelta(years=1)).strftime('%Y-%m-%d')
        Model_Year[i1] = (Model_Start, Model_End)
        
        for i2 in range(0,n_utilities):
            # For each utility find regression models
            All_Models = EnPI_LinearRegression(Model_Energy.loc[Model_yr_idxs[i1]].iloc[:,i2], Model_Vars.loc[Model_yr_idxs[i1]], Model_Year[i1], Model_Energy.columns[i2])
            All_Models = All_Models.loc[All_Models['Valid?'] == True]
            
            # Ouput warning that there are no value models for utility in this year
            if All_Models.empty:
                print('\n\tNo Valid Model for "' + Model_Energy.columns[i2] + '" with model year starting ' + Model_Dates[Model_yr_idxs[i1]].iloc[0].strftime('%Y-%m-%d'))

            # Still add data to the list of all models even if empty to avoid out of index warning!
            EnPI_Models.append(All_Models)
        
        # Add results to the list of models
        EnPI_Valid_Models[i1] = EnPI_Models
        
    del All_Models, EnPI_Models, i1, i2, i
    
    
    
    # ***** ERROR CHECK ***** #
    
    Valid_Model_Check = list(range(0,len(EnPI_Valid_Models)))
    
    for i in range(0, len(EnPI_Valid_Models)):
        if len(EnPI_Valid_Models[i]) != n_utilities:
            Valid_Model_Check.remove(i)
            
    
    
    if Valid_Model_Check == []:
        print('')
        print('********************** ERROR **********************')
        print('NO MODEL YEAR WITH VALID MODELS FOR ALL UTILITIES!')
        print('***************************************************')
        print('\n')
        print('Solution:')
        print('Rerun program using Energy Intensity method.')
        print('- OR -')
        print('Try new relevant variables in the regression.')
        
        EnPI_Valid_Model
    
        raise StopExecution
    
    # If you make it here, there is a year with valid models for all utilities
    # Pick model if one has not been specified
    
    # Display results of model fit but don't do analysis
    if Show_Models:
        
        print('EnPI is in display regression fit data mode...')
        
        for i1 in range(0,len(EnPI_Valid_Models)):
            print('\nModel Data for Model Year: ' + Model_Year[i1][0] + ' --> ' + Model_Year[i1][1])
            
            for i2 in range(0,n_utilities):
                display(EnPI_Valid_Models[i1][i2][['Utility', 'Variables', 'Adjusted R2', 'Coefficients']])
            
        raise StopExecution
    
    # If show model fit data is false, do the EnPI analysis
    else:
        
        # Create an array to hold the locations of the highest adjusted R2 values for all utilities and model years
        Model_idxs = np.zeros((n_model_years,n_utilities))
        
        # If the model range is greater than 1 year, pick the year with the best total adjusted R2 value
        if (Model_Range_Start + relativedelta(years=1)) != Model_Range_End:
        
            # Create empty lists to hold information about the models in each model year
            Adjusted_R2s = [0]*n_model_years
            
            # Cycle through all model years
            for i1 in range(0,len(EnPI_Valid_Models)):
                # Cycle through each utility
                for i2 in range(0,n_utilities):
                    
                    # Find the index of the utility model that has the highest adjusted R2
                    Model_idxs[i1][i2] += EnPI_Valid_Models[i1][i2]['Adjusted R2'].idxmax()
                    
                    # Find the value of the total adjusted R2 for all models
                    Adjusted_R2s[i1] += EnPI_Valid_Models[i1][i2]['Adjusted R2'][Model_idxs[i1][i2]]
                    
                print('\tTotal Adjusted R2 for Model Year ' + Model_Year[i1][0] + ' --> ' + Model_Year[i1][1] + ' is: ' + f'{Adjusted_R2s[i1]:.3f}')
    
            # Select the model year and the models for each utility
            Selected_Year = np.argmax(Adjusted_R2s)
            
            print('\n\tProgram has selected ' + Model_Year[Selected_Year][0] + ' --> ' + Model_Year[Selected_Year][1] + ' as the model year!\n')
        
        else:
            
            # There is only 1 model year
            Selected_Year = 0
            
            # Get the index of the utility model with the highest adjusted R2 values
            for i1 in range(0,n_utilities):
                Model_idxs[0][i1] += EnPI_Valid_Models[0][i1]['Adjusted R2'].idxmax()
            
            print('\n\tYou have selected ' + Model_Year[Selected_Year][0] + ' --> ' + Model_Year[Selected_Year][1] + ' as the model year!\n')
        
        # If no models are specified, pick the ones with the best adjusted R2
        if Chosen_Models == []:
            Chosen_Models = Model_idxs[Selected_Year,:].astype(int)
        
        # Create empty list to hold the final selected models
        EnPI_Final_Models = list()
        flag = False
        
        # Get the final variables for the model
        Final_Vars = Model_Vars.loc[:,EnPI_Valid_Models[Selected_Year][0].loc[Chosen_Models[0],'Variables']]
        
        # Check for co-linearity using VIFs if more than one variable
        if Final_Vars.shape[1] > 1:
            
            vif = pd.DataFrame()
            for i in range(Final_Vars.shape[1]):
                vif.loc[i,'VIF Factor'] = variance_inflation_factor(Final_Vars.values,i)
            
            vif['features'] = Final_Vars.columns
        
        
        # Get the final model information
        for i1 in range(0,n_utilities):
            
            # Check if model at specified index exists
            if Chosen_Models[i1] in EnPI_Valid_Models[Selected_Year][i1].index:
                
                # Append model to list of final models
                EnPI_Final_Models.append(EnPI_Valid_Models[Selected_Year][i1].loc[Chosen_Models[i1],:])
                
                # Print the variables used for the specified utility model
                print('\tVariables for ' + EnPI_Final_Models[i1]['Utility'] + ' Model (index = ' + str(Chosen_Models[i1]) + ') are: ' + ', '.join(EnPI_Final_Models[i1].Variables))
                
            else:
                
                print('******** ERROR ********')
                print('SPECIFIED MODEL FOR: ' + Model_Energy.columns[i1] + ' DOES NOT EXIST!')
                print('***********************\n')
                
                flag = True
                
        if flag:
            print('Solution:')
            print('Please check the specified models.')
            print('Data for models in selected year:\n')
            
            for i1 in range(0,n_utilities):
                display(EnPI_Valid_Models[Selected_Year][i1][['Utility', 'Variables', 'Adjusted R2', 'Coefficients']])
            
            raise StopExecution
                

        Model_Dates = Model_Dates.loc[Model_yr_idxs[Selected_Year]]
        Model_Energy = Model_Energy.loc[Model_yr_idxs[Selected_Year]]
        Model_Vars = Model_Vars.loc[Model_yr_idxs[Selected_Year]]   
    
    
    
    # ***** PLOT INFORMATION ON SELECTED RELEVANT VARIABLES ***** #
        
    EnPI_PlotModelVariables(Model_Energy, Model_Vars, n_utilities, n_samples)
    
    
    
    # ***** MAKE PLOT FOR MODEL FIT ***** #
        
    plt.figure()
    plt.gca()
    
    # Plot all of the model data first
    x = range(1, len(Model_Dates)+1)
    for i in range(0,n_utilities):
        plt.plot(x,Model_Energy.iloc[:,i],linestyle='none',marker='o')
    
    # Plot all of the fits next
    Fit_Legend = []
    for i in range(0,n_utilities):
        plt.plot(x,EnPI_Final_Models[i]['Model'].predict())
        Fit_Legend.append(Model_Energy.columns[i][0:Model_Energy.columns[0].find('(')-1] + ' Model')
    
    plt.legend(np.concatenate((Model_Energy.columns, Fit_Legend)))
    plt.ylabel('Energy (MMBtu)')
    plt.title('Model Year')       
    plt.grid() 
    
    
    
    # ***** CHECK REPORTING YEAR DATA VAILIDITY ***** #

    print('\nChecking Model Data Validity...\n')

    for i in range(0,len(EnPI_Final_Models)):
        
        cols = EnPI_Final_Models[i]['Variables']
        Data_Check = SEP_DataValidityCheck(Model_Vars[cols], Reporting_Vars[cols],n_samples)
        flag = False
        
        if any(Data_Check == False):
            print('******** ERROR ********')
            print('DATA FOR: ' + Model_Energy.columns[i] + ' FAILS VAILIDTY TEST!')
            print('***********************\n')
                        
            flag = True
        else:
            print('\t' + Model_Energy.columns[i] + ' : PASS')
            
        if flag:
            print('Solution:')
            print('Try a different model year or ask TAM about banking approach.')
            print('- OR -')
            print('Try new relevant variables in the regression.')
            
            raise StopExecution
    
    del cols
      
    
    
    # ***** DO ENPI SAVINGS CALCULATIONS ***** #

    Model_yr_idx = (Model_Dates.iloc[0] <= Reporting_Dates) & (Reporting_Dates < Model_Dates.iloc[0] + relativedelta(years=1))
    (EnPI_Results, Simulated_Energy) = EnPI_FacilitySavings(EnPI_Final_Models, Reporting_Energy, Reporting_Vars, Model_yr_idx, n_utilities, n_years, n_samples)   
    
    
    
    # ***** PLOT SIMULATION RESULTS ***** #

    Reporting_Range = [Reporting_Range_Start, Reporting_Range_End]
    Model_Range = [Model_Dates.iloc[0], Model_Dates.iloc[0] + relativedelta(years=1)]

    EnPI_PlotResults(Reporting_Energy, Simulated_Energy, Reporting_Dates, Model_yr_idx, n_utilities, n_years, n_samples, Reporting_Range,  Model_Range)
    
    
    
    # ***** ENERGY INTENSITY IMPROVEMENT PLOT ***** #

    plt.figure()
    gca = plt.gca()
    
    plt.plot(range(1,n_years+1), EnPI_Results.loc['Annual Improvement in Energy Intensity (%)'].values, marker='o', markersize=5, linewidth=2)
    plt.plot(range(1,n_years+1), EnPI_Results.loc['Total Improvement in Energy Intensity (%)'].values, marker='x', markersize=7.5, linewidth=2, markeredgewidth=2)
    
    gca.set_xlabel('Reporting Year')
    gca.set_xticks(range(1,n_years + 1))
    gca.set_xlim([0.5, n_years+0.5])
    
    gca.set_ylabel('Percent Improvement')
    gca.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    
    plt.legend(['Annual Improvement in EI', 'Total Improvement in EI'],loc=2,fontsize=9)
    plt.grid()
    
    box = gca.get_position()
    gca.set_position([box.x0-0.02, box.y0, box.width, box.height])
    
    return EnPI_Results
    

#%% EnPI Linear Regression 
def EnPI_LinearRegression(mdl_data, mdl_vars, mdl_range, header):
    
    # Get number of relevant variables
    num_vars = np.shape(mdl_vars)[1]
    num_points = np.shape(mdl_vars)[0]
    
    # Get all permutations of relevant variables
    C = []
    idx = range(0,num_vars)
    for i in idx:
        C += list(combinations(idx, i+1))
        
    # Create a dataframe object to store regression data
    EnPI_Models = pd.DataFrame(columns=["Utility", "Model Range", "Model", "Valid?", "Variables", "Variable p-Values", "R2", "Adjusted R2", "Model p-Value", "Coefficients"])
        
    # Find regression models for each combination of relevant variables
    for i in range(0,len(C)):
        X = mdl_vars.iloc[:,np.asarray(C[i])]
        
        X = np.append(X,np.ones((num_points,1)),axis=1)
        mdl = sm.OLS(mdl_data,X).fit()
        
        new_row = pd.Series([header, 
                             mdl_range,
                             mdl,SEP_ModelValidityCheck(mdl),
                             mdl_vars.columns[np.asarray(C[i])].values, 
                             np.around(mdl.pvalues[0:-1].values,5), 
                             mdl.rsquared, 
                             mdl.rsquared_adj, 
                             mdl.f_pvalue, 
                             np.around(mdl.params.values,5)], 
                            index=EnPI_Models.columns)
        
        EnPI_Models.loc[i+1] = new_row
        
    return EnPI_Models


# Checks if reporting data is within 3 standard deviatoins of the model year
def SEP_DataValidityCheck(Model_Vars, Reporting_Vars, n):
    
    Std = np.std(Model_Vars, axis=0)
    Mean = np.mean(Model_Vars, axis = 0)
    
    # Find average for each reporting year    
    Vars_Avg = Reporting_Vars.groupby(Reporting_Vars.index // n).sum() / n
    
    # Variable relivance test
    isValid_1 = (np.min(Model_Vars, axis=0) < Vars_Avg) & (Vars_Avg < np.max(Model_Vars, axis=0))
    isValid_2 = (Mean-3*Std < Vars_Avg) & (Vars_Avg < Mean+3*Std)
        
    isValid = isValid_1.all() | isValid_2.all()
    
    return isValid

# Checks if a model is valid based on SEP criteria
def SEP_ModelValidityCheck(EnPI_Model):
    
    isValid = True
    
    isValid &= EnPI_Model.f_pvalue < 0.1
    isValid &= all(EnPI_Model.pvalues[:-1] < 0.2)
    isValid &= any(EnPI_Model.pvalues[:-1] < 0.1)
    isValid &= EnPI_Model.rsquared >= 0.5
    
    return isValid

# Makes plots of the model variables
def EnPI_PlotModelVariables(Model_Energy, Model_Vars, n_utilities, n):
    
    sns.pairplot(Model_Vars, kind="reg")
    
    axs = np.asarray(plt.gcf().get_axes())
    axs = axs[np.arange(0,len(axs),len(Model_Vars.columns)+1)]
    
    for i in range(0,len(axs)):
        axs[i].set_visible(False)
    
    
    x = range(1,n+1)

    for i in range(0,Model_Vars.shape[1]):
        fig, axs = plt.subplots(n_utilities,2)
        
        axs[0,0].plot(x,Model_Vars.iloc[:,i])
        axs[0,0].title.set_text(Model_Vars.columns[i])
        plt.sca(axs[0,0])
        plt.xlim([0,n+2])
        
        for j in range(0,n_utilities):
            if j > 0:
                axs[j,0].set_visible(False)
                
            axs[j,1].plot(Model_Vars.iloc[:,i],Model_Energy.iloc[:,j-1],marker='o',linestyle='',markersize=3)
            axs[j,1].set_xlim((0,axs[j,1].get_xlim()[1]))
            axs[j,1].set_ylim((0,axs[j,1].get_ylim()[1]))
            plt.sca(axs[j,1])
            plt.ylabel(Model_Energy.columns[j-1])
            plt.xlabel(Model_Vars.columns[i])
            
        fig.tight_layout()


# Do EnPI Savings Calculations
def EnPI_FacilitySavings(EnPI_Valid_Models, Reporting_Energy, Reporting_Vars, Model_yr_idx, n_utilities, n_years, n_samples):
    
    # Create empty dataframe for simulated energy
    Simulated_Energy = pd.DataFrame(columns=Reporting_Energy.columns, index=Reporting_Energy.index)

    # Simulate energy during the reporting period
    for i in range(0,n_utilities):
        Simulation_Vars = EnPI_Valid_Models[i]['Variables']
        Simulation_Input = np.append(Reporting_Vars[Simulation_Vars], np.ones((len(Reporting_Vars),1)), axis=1)
        Simulated_Energy.iloc[:,i] = EnPI_Valid_Models[i]['Model'].predict(Simulation_Input)
    
    # Create row labels for EnPI Results dataframe
    EnPI_Results_Indices = list('Actual ' + Reporting_Energy.columns)
    EnPI_Results_Indices.append('TOTAL ACTUAL ENERGY (MMBtu)')
    EnPI_Results_Indices += list('Modeled ' + Reporting_Energy.columns)
    EnPI_Results_Indices.append('TOTAL MODELED ENERGY (MMBtu)')
    EnPI_Results_Indices.append('Adjustment Method')
    EnPI_Results_Indices += list(Reporting_Energy.columns + ' Annual Savings')
    EnPI_Results_Indices.append('New Energy Savings for Current Year (MMBtu/Year)')
    EnPI_Results_Indices.append('Cumulative Savings (MMBtu)')
    EnPI_Results_Indices.append('TOTAL ENERGY SAVINGS SINCE BASELINE YEAR (MMBtu/Year)')
    EnPI_Results_Indices.append('Total Improvement in Energy Intensity (%)')
    EnPI_Results_Indices.append('Annual Improvement in Energy Intensity (%)')
    EnPI_Results_Indices.append('Adjustment for Baseline Primary Energy Use (MMBtu/Year)')
    
    # Create EnPI Results like table
    EnPI_Results = pd.DataFrame(columns=range(1,n_years+1), index = EnPI_Results_Indices)
    
    # Fill in Actual Energy Usage
    EnPI_Results.iloc[0:n_utilities] = Reporting_Energy.groupby(Reporting_Vars.index // n_samples).sum().transpose().values
    EnPI_Results.iloc[n_utilities] = EnPI_Results.iloc[0:n_utilities].sum()
    EnPI_Results.iloc[n_utilities+1:2*n_utilities+1] = Simulated_Energy.groupby(Simulated_Energy.index // n_samples).sum().transpose().values
    EnPI_Results.iloc[2*n_utilities+1] = EnPI_Results.iloc[n_utilities+1:2*n_utilities+1].sum()
    
    # Set baseline year savings to be 0
    EnPI_Results.iloc[-8:,0] = 0
    
    print('\n\nCalculation EnPI Results... ')
    print('Regression Method is: ', end='')
    
    # Forecasting (if the first element of baseline index is True)
    if (Model_yr_idx.iloc[0] == True) | (not any(Model_yr_idx)):
        print('Forecasting')
    
        EnPI_Results.loc['Adjustment Method'].iloc[:] = 'ForeCast'
        
        if any(Model_yr_idx):
            EnPI_Results.loc['Adjustment Method'].iloc[0] = 'Model Year'
        
        EnPI_Results.iloc[-n_utilities-6:-6,1:] = EnPI_Results.iloc[n_utilities+1:2*n_utilities+1,1:].values - EnPI_Results.iloc[0:n_utilities,1:].values
    
        SEnPIs = EnPI_Results.loc['TOTAL ACTUAL ENERGY (MMBtu)']/EnPI_Results.loc['TOTAL MODELED ENERGY (MMBtu)']
        
        EnPI_Results.loc['Total Improvement in Energy Intensity (%)'][1:] = (1 - SEnPIs[1:]) * 100    
        EnPI_Results.loc['Annual Improvement in Energy Intensity (%)'][1:] = np.diff(EnPI_Results.loc['Total Improvement in Energy Intensity (%)'])
        
        EnPI_Results.loc['TOTAL ENERGY SAVINGS SINCE BASELINE YEAR (MMBtu/Year)'] = EnPI_Results.iloc[2*n_utilities+1] - EnPI_Results.iloc[n_utilities]
        
        
    # Backcasting (if the last element of baseline index is True)
    elif Model_yr_idx.iloc[-1] == True:
        print('Backcasting')
        
        EnPI_Results.loc['Adjustment Method'].iloc[-1] = 'Model Year'
        EnPI_Results.loc['Adjustment Method'].iloc[:-1] = 'Backcast'
        
        SEnPIs = EnPI_Results.loc['TOTAL MODELED ENERGY (MMBtu)'].values/EnPI_Results.loc['TOTAL ACTUAL ENERGY (MMBtu)'].values
        EnPI_Results.loc['Annual Improvement in Energy Intensity (%)'].iloc[1:] = np.diff(SEnPIs)*100
        
        # Last years annual energy saved + (last year actual - last year modeled) - (this year actual - this year modeled)
        for i in range(1,EnPI_Results.shape[1]):
            EnPI_Results.iloc[-(n_utilities+6):-6,i] = EnPI_Results.iloc[-(n_utilities+6):-6,i-1] + (
                EnPI_Results.iloc[0:n_utilities,i-1].values - EnPI_Results.iloc[n_utilities+1:2*n_utilities+1,i-1].values) - (
                EnPI_Results.iloc[0:n_utilities,i].values - EnPI_Results.iloc[n_utilities+1:2*n_utilities+1,i].values)
           
            EnPI_Results.loc['Total Improvement in Energy Intensity (%)'].iloc[i] = EnPI_Results.loc['Annual Improvement in Energy Intensity (%)'].iloc[i] + EnPI_Results.loc['Total Improvement in Energy Intensity (%)'].iloc[i-1]
            
            EnPI_Results.loc['TOTAL ENERGY SAVINGS SINCE BASELINE YEAR (MMBtu/Year)'].iloc[i] = (
                EnPI_Results.loc['TOTAL ENERGY SAVINGS SINCE BASELINE YEAR (MMBtu/Year)'].iloc[i-1] - (
                np.diff(EnPI_Results.iloc[[n_utilities,2*n_utilities+1],i-1].values).item() - (
                np.diff(EnPI_Results.iloc[[n_utilities,2*n_utilities+1],i].values).item() )))
        
    # Chaining
    else:
        print('Chaining')
        
        mdl_idx = (Model_yr_idx.groupby(Model_yr_idx.index // n_samples).sum() != 0).values
        mdl_idx = np.where(mdl_idx)[0].item()
        
        EnPI_Results.loc['Adjustment Method'].iloc[:] = 'Chaining'
        EnPI_Results.loc['Adjustment Method'].iloc[mdl_idx] = 'Model Year'
        
        SEnPIs = EnPI_Results.loc['TOTAL MODELED ENERGY (MMBtu)'].values/EnPI_Results.loc['TOTAL ACTUAL ENERGY (MMBtu)'].values
        SEnPIs[mdl_idx+1:] = SEnPIs[0]/SEnPIs[mdl_idx+1:]
        
        for i in range(1,EnPI_Results.shape[1]):
            if i <= mdl_idx:
                EnPI_Results.iloc[-(n_utilities+6):-6,i] = EnPI_Results.iloc[-(n_utilities+6):-6,i-1] + (
                    EnPI_Results.iloc[0:n_utilities,i-1].values - EnPI_Results.iloc[n_utilities+1:2*n_utilities+1,i-1].values) - (
                    EnPI_Results.iloc[0:n_utilities,i].values - EnPI_Results.iloc[n_utilities+1:2*n_utilities+1,i].values )
                
                EnPI_Results.loc['Annual Improvement in Energy Intensity (%)'].iloc[i] = (SEnPIs[i] - SEnPIs[i-1])*100
                
                EnPI_Results.loc['Total Improvement in Energy Intensity (%)'].iloc[i] = EnPI_Results.loc['Annual Improvement in Energy Intensity (%)'].iloc[i] + EnPI_Results.loc['Total Improvement in Energy Intensity (%)'].iloc[i-1]
                
                EnPI_Results.loc['TOTAL ENERGY SAVINGS SINCE BASELINE YEAR (MMBtu/Year)'].iloc[i] = (
                    EnPI_Results.loc['TOTAL ENERGY SAVINGS SINCE BASELINE YEAR (MMBtu/Year)'].iloc[i-1] - (
                    np.diff(EnPI_Results.iloc[[n_utilities,2*n_utilities+1],i-1].values).item() - (
                    np.diff(EnPI_Results.iloc[[n_utilities,2*n_utilities+1],i].values).item() )))
                
            else:
                EnPI_Results.iloc[-(n_utilities+6):-6,i] = EnPI_Results.iloc[n_utilities+1:2*n_utilities+1,i].values - EnPI_Results.iloc[0:n_utilities,i].values
                EnPI_Results.loc['Total Improvement in Energy Intensity (%)'].iloc[i] = (1 - SEnPIs[i]) * 100
                EnPI_Results.loc['Annual Improvement in Energy Intensity (%)'].iloc[i] = np.diff(EnPI_Results.loc['Total Improvement in Energy Intensity (%)'].iloc[i-1:i+1]).item()
                EnPI_Results.loc['TOTAL ENERGY SAVINGS SINCE BASELINE YEAR (MMBtu/Year)'].iloc[i] = (
                    EnPI_Results.loc['TOTAL MODELED ENERGY (MMBtu)'].iloc[i] - EnPI_Results.loc['TOTAL ACTUAL ENERGY (MMBtu)'].iloc[i] )
    
    # Final common calculations
    EnPI_Results.loc['New Energy Savings for Current Year (MMBtu/Year)'][1:] = np.diff(EnPI_Results.loc['TOTAL ENERGY SAVINGS SINCE BASELINE YEAR (MMBtu/Year)'])
    EnPI_Results.loc['Cumulative Savings (MMBtu)'] = EnPI_Results.loc['TOTAL ENERGY SAVINGS SINCE BASELINE YEAR (MMBtu/Year)'].cumsum()
    EnPI_Results.loc['Adjustment for Baseline Primary Energy Use (MMBtu/Year)'] = EnPI_Results.loc['TOTAL ACTUAL ENERGY (MMBtu)'] + EnPI_Results.loc['TOTAL ENERGY SAVINGS SINCE BASELINE YEAR (MMBtu/Year)'] - EnPI_Results.loc['TOTAL ACTUAL ENERGY (MMBtu)'][1]
    
    return EnPI_Results, Simulated_Energy
        

# Plot EnPI Regression Results
def EnPI_PlotResults(Reporting_Energy, Simulated_Energy, Reporting_Dates, Model_yr_idx, n_utilities, n_years, n_samples, Reporting_Range,  Model_Range):
    plt.figure()
    axs = plt.gca()
    
    for i in range(0,n_utilities):
        plt.plot(Reporting_Dates, Reporting_Energy.iloc[:,i],linestyle='none',marker='o')
        
        # Plot all of the fits next
    Simulation_Legend = []
    for i in range(0,n_utilities):
        if not all(Simulated_Energy.iloc[:,0] == 0):
            plt.plot(Reporting_Dates,Simulated_Energy.iloc[:,i])
            Simulation_Legend.append(Simulated_Energy.columns[i][0:Simulated_Energy.columns[i].find('(')-1] + ' Model')
    
    plt.xticks(Reporting_Dates[0] + np.arange(n_years + 1)*relativedelta(months=12))
    plt.xlim([Reporting_Range[0], Reporting_Range[-1]])
    plt.ylabel('Energy (MMBtu)')
    plt.legend(np.concatenate((Reporting_Energy.columns, Simulation_Legend)))
    plt.grid()
    
    ylim = np.asarray(plt.ylim())
    # plt.plot([Model_Range_Start]*2,ylim,linestyle='--',linewidth=1.5,color='k')
    # plt.plot([Model_Range_End]*2,ylim,linestyle='--',linewidth=1.5,color='k')
    
    
    ## Place arrows at this height over the top of the axis
    y = 1.02
    
    # Plot Baseline Range
    x_Baseline = [Reporting_Dates.iloc[0], Reporting_Dates[n_samples]]
    x_Baseline = np.divide(np.subtract(x_Baseline,Reporting_Dates.iloc[0]),Reporting_Range[-1] - Reporting_Dates[0])
    
    axs.annotate('', xy=(x_Baseline[0], y), xycoords='axes fraction', xytext=(x_Baseline[1], y), 
                arrowprops=dict(arrowstyle="<|-|>", color='b'))
    
    # Plot Reporting Year Range
    x_Reporting = [Reporting_Dates.iloc[-n_samples], Reporting_Range[-1]]
    x_Reporting = np.divide(np.subtract(x_Reporting,Reporting_Dates.iloc[0]),Reporting_Range[-1] - Reporting_Dates[0])
    
    axs.annotate('', xy=(x_Reporting[0], y), xycoords='axes fraction', xytext=(x_Reporting[1], y), 
                arrowprops=dict(arrowstyle="<|-|>", color='b'))
    
    # Plot arrow for model year if not the same as the baseline or reporting range
    if (Model_yr_idx.iloc[-1] == False) & (Model_yr_idx.iloc[0] == False):
        x_Model = [Model_Range[0], Model_Range[-1]]
        x_Model = np.divide(np.subtract(x_Model,Reporting_Dates.iloc[0]),Reporting_Range[-1] - Reporting_Dates[0])
        
        axs.annotate('', xy=(x_Model[0], y), xycoords='axes fraction', xytext=(x_Model[1], y), 
                    arrowprops=dict(arrowstyle="<|-|>", color='b'))    
    
    
    ## Add labels to the ranges depending on regression method
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    txt = ['Baseline','Reporting','Model']
    
    x = []
    y = 1.05
    
    # Forecasting
    if Model_yr_idx.iloc[0] == True:
        del txt[2]
        txt[0] += ' & Model Year'
        txt[1] += ' Year'
        
        x.append(x_Baseline)
        x.append(x_Reporting)
        
    # Backcasting
    elif Model_yr_idx.iloc[-1] == True:
        del txt[2]
        txt[0] += ' Year'
        txt[1] += ' & Model Year'
        
        x.append(x_Baseline)
        x.append(x_Reporting)
    
    # Forecast with baseline outside reporting range
    elif not any(Model_yr_idx):
        del txt[2]
        
        x.append(x_Baseline)
        x.append(x_Reporting)
        
    # Chaining
    else:
        txt[0] += ' Year'
        txt[1] += ' Year'
        txt[2] += ' Year'
        
        x.append(x_Baseline)
        x.append(x_Reporting)
        x.append(x_Model)
    
    for i in range(0,len(txt)):
        axs.text(np.mean(x[i]), y, txt[i], ha="center", va="bottom", 
                 size=10, bbox=bbox_props, transform=axs.transAxes)
    
    plt.ylim(ylim)
    
    
    
    ##### Create a total energy plot like in EnPI #####
    plt.figure()
    axs = plt.gca()
    
    
    # Plot actual data
    h1, = plt.plot(Reporting_Dates, Reporting_Energy.sum(axis=1),linestyle='none',marker='o', markersize=5)
        
    # Plot modeled data
    h2, = plt.plot(Reporting_Dates, Simulated_Energy.sum(axis=1),linestyle='-', marker='')
    
    # Formatting
    plt.xticks(Reporting_Dates[0] + np.arange(n_years + 1)*relativedelta(months=12))
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xlim([Reporting_Range[0], Reporting_Dates[0] + relativedelta(years=n_years)])
    plt.ylabel('Total Energy (MMBtu)', fontsize=9)
    plt.grid()
    
    ylim = np.asarray(plt.ylim())
    plt.ylim(ylim)
    
    ## Place arrows at this height over the top of the axis
    y = 1.04
    
    # Plot Baseline Range
    x_Baseline = [Reporting_Dates.iloc[0], Reporting_Dates[n_samples]]
    x_Baseline = np.divide(np.subtract(x_Baseline,Reporting_Dates.iloc[0]),Reporting_Range[-1] - Reporting_Dates[0])
    
    axs.annotate('', xy=(x_Baseline[0], y), xycoords='axes fraction', xytext=(x_Baseline[1], y), 
                arrowprops=dict(arrowstyle="<|-|>", color='b'))
    
    # Plot Reporting Year Range
    x_Reporting = [Reporting_Dates.iloc[-n_samples], Reporting_Range[-1]]
    x_Reporting = np.divide(np.subtract(x_Reporting,Reporting_Dates.iloc[0]),Reporting_Range[-1] - Reporting_Dates[0])
    
    axs.annotate('', xy=(x_Reporting[0], y), xycoords='axes fraction', xytext=(x_Reporting[1], y), 
                arrowprops=dict(arrowstyle="<|-|>", color='b'))
    
    # Plot arrow for model year if not the same as the baseline or reporting range
    if (Model_yr_idx.iloc[-1] == False) & (Model_yr_idx.iloc[0] == False):
        x_Model = [Model_Range[0], Model_Range[-1]]
        x_Model = np.divide(np.subtract(x_Model,Reporting_Dates.iloc[0]),Reporting_Range[-1] - Reporting_Dates[0])
        
        axs.annotate('', xy=(x_Model[0], y), xycoords='axes fraction', xytext=(x_Model[1], y), 
                    arrowprops=dict(arrowstyle="<|-|>", color='b'))    
    
    
    ## Add labels to the ranges depending on regression method
    bbox_props = dict(boxstyle="round", fc='None', ec='None', alpha=0.9)
    txt = ['Baseline','Reporting','Model']
    
    x = []
    y = 1.06
    
    # Forecasting
    if Model_yr_idx.iloc[0] == True:
        del txt[2]
        txt[0] += ' & Model Year'
        txt[1] += ' Year'
        
        x.append(x_Baseline)
        x.append(x_Reporting)
        
    # Backcasting
    elif Model_yr_idx.iloc[-1] == True:
        del txt[2]
        txt[0] += ' Year'
        txt[1] += ' & Model Year'
        
        x.append(x_Baseline)
        x.append(x_Reporting)
    
    # Forecast with baseline outside reporting range
    elif not any(Model_yr_idx):
        del txt[2]
        
        x.append(x_Baseline)
        x.append(x_Reporting)
        
    # Chaining
    else:
        txt[0] += ' Year'
        txt[1] += ' Year'
        txt[2] += ' Year'
        
        x.append(x_Baseline)
        x.append(x_Reporting)
        x.append(x_Model)
    
    for i in range(0,len(txt)):
        axs.text(np.mean(x[i]), y, txt[i], ha="center", va="bottom", 
                 size=9, bbox=bbox_props, transform=axs.transAxes)

    # Adjust font size for both axis tick labels
    axs.tick_params(axis='both', which='major', labelsize=9)
    
    # Make second y axis use comma seperator
    axs.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    
    
    ##### FORMATTING JUST FOR THE SUMMARY GUIDE #####
    
    # # Plot Baseline Range
    # x_Baseline = Reporting_Dates[n_samples]
    # x_Baseline = np.divide(np.subtract(x_Baseline,Reporting_Dates.iloc[0]),Reporting_Range[-1] - Reporting_Dates[0])
    
    # axs.annotate('', xy=(x_Baseline, -0.05), xytext = (x_Baseline, y), xycoords='axes fraction',
    #             arrowprops=dict(arrowstyle='-', linestyle='--', color='k', linewidth=1.5))
    
    # Total_Reporting_Energy = Reporting_Energy.sum(axis=1)
    # Total_Simulated_Energy = Simulated_Energy.sum(axis=1)
    
    # x = Reporting_Dates[n_samples:]
    # x = x.append(x[::-1])
    # x = x.to_numpy().reshape(-1,1).astype(float)
    # x = x/1E9/3600/24 + 719163
    
    # y = np.append(Total_Reporting_Energy[n_samples:], Total_Simulated_Energy[:n_samples-1:-1]).reshape(-1,1)
    
    # h3 = patches.Polygon(np.concatenate((x,y), axis=1), label='Energy Savings')
    # plt.setp(h3,facecolor='g',alpha=0.2)

    # axs.add_patch(h3)

    # plt.legend([h1,h2,h3],['Actual Energy','Modeled Energy','Energy Savings'],ncol=1, fontsize=8, loc='upper right')
    
    

#%% ##### CORPORATE ROLL UP CODE ##### %%#
def EnPI_RollUp_GetFiles(Select_Mode, folder_loc, file_names):
    
    if Select_Mode == 'Select':
        
        # Open a dialog window to get the file names
        root = Tk()
        root.withdraw()
        root.title('EnPI Corporate Roll Up Tool')
        file_names = filedialog.askopenfilenames(title='EnPI Corporate Roll Up', filetypes=[('json files','*.json')])
    
    
    elif Select_Mode == 'List':
        
        # Cycle through each listed file
        for i in range(0,len(file_names)):
            
            # Check if the files have .json extensions and add if not
            if not(file_names[i].endswith('.json')):
                file_names[i] += '.json'
                
            # If the folder location is not empty, add extension
            if folder_loc:
                file_names[i] = folder_loc + '/' + file_names[i]
        
    else:
        print('*********************** ERROR ************************')
        print('UNRECOGNIZED MODE: ' + Select_Mode + ' IS NOT AN OPTION!')
        print('******************************************************\n')
                    
        print('Solution:')
        print('Please check the specified mode for file selection.\n')
                
        raise StopExecution
        
    # Check if file_names is empty
    if not file_names:
        print('****** ERROR ******')
        print('FILE LIST IS EMPTY!')
        print('*******************\n')
                    
        print('Solution:')
        print('Please check roll up mode and your list of files.\n')
                
        raise StopExecution
        
        
    flag = False
    facility_results = []
    
    # Cycle through each listed file to  make sure it exists
    for i in range(0,len(file_names)):
        
        if not(path.exists(file_names[i])):
            flag = True
            print('ERROR: ' + file_names[i] + 'does not exist!')
    
    # If a file does not exist, then issue a error and exit the program
    if flag:
        print('*********************** ERROR ***********************')
        print('MISSING FILE: AT LEAST ONE LISTED FILE DOES NOT EXIST!')
        print('*****************************************************\n')
                    
        print('Solution:')
        print('Please check the specified folder and file names.\n')
                
        raise StopExecution
    
    # If make it here, then all the files exist
    for i in range(0,len(file_names)):
        
        print('\nLoading: ' + file_names[i][file_names[0].rfind('/')+1:])
        
        # Load the json file into python as a pandas dataframe    
        facility_results.append(pd.read_json(file_names[i]))
        
        # Print the dataframe to the screen
        display(facility_results[i])

    return facility_results



def EnPI_RollUp_Calculation(facility_results):

    # ***** CREATE CORPORATE ROLL UP RESULTS TABLE ***** #
    
    # Get a list of all utilities in the json files...
    Energy_Types = []
    
    # Run through utilities of each facility
    for i in range(0,len(facility_results)):
        n_util = facility_results[0].index.get_loc('TOTAL ACTUAL ENERGY (MMBtu)')
        utils = facility_results[i].index[0:n_util]
        Energy_Types.append([x[7:-8].title() for x in utils])
    
    # Get list of unique energy types    
    Energy_Types = np.unique(Energy_Types)
    
    # Define rows for roll up dataframe
    EnPI_Results_Indices = list(['Actual ' + x + ' (MMBtu)'for x in Energy_Types])
    EnPI_Results_Indices.append('TOTAL PRIMARY ENERGY CONSUMED (MMBtu)')
    EnPI_Results_Indices.append('Adjustment for Baseline Primary Energy Use (MMBtu/Yr)')
    EnPI_Results_Indices.append('Adjusted Baseline Primary Energy Use (MMBtu/Yr)')
    EnPI_Results_Indices.append('Annual Improvement in Energy Intensity (%)')
    EnPI_Results_Indices.append('Total Improvement in Energy Intensity (%)')
    EnPI_Results_Indices.append('New Energy Savings for Current Year (MMBtu/Year)')
    EnPI_Results_Indices.append('TOTAL ENERGY SAVINGS SINCE BASELINE YEAR (MMBtu/Year)')
    
    # Get the number of years in the roll up report
    n_years = facility_results[0].columns.size
    
    # Create EnPI Results like table
    EnPI_Results = pd.DataFrame(columns=range(1,n_years+1), index = EnPI_Results_Indices)   
    
    # Create empty variables to fill with facility data
    Actual_Energy = np.zeros((len(Energy_Types),n_years))
    New_Savings = np.zeros((1,n_years))
    Total_Savings = np.zeros((1,n_years))
    Total_Improve = 0
        
    # Fill the data fram with totals for each facility
    for i in range(0,len(facility_results)):
        
        # Check each facility for that energy type:            
        for j in range(0,len(Energy_Types)):
            
            # Get the number and name of the utilies for this facility
            n_util = facility_results[0].index.get_loc('TOTAL ACTUAL ENERGY (MMBtu)')
            utils = facility_results[i].index[0:n_util]
            
            # Check each utility to see if label contains energy type
            for k in range(0,len(utils)):
                if Energy_Types[j].lower() == utils[k][7:-8].lower():
                    Actual_Energy[j,:] = np.add(Actual_Energy[j,:],facility_results[i].iloc[k,:].values)
                    
        # Get for this facility
        New_Savings = np.add(New_Savings, facility_results[i].loc['New Energy Savings for Current Year (MMBtu/Year)'].values)
        Total_Savings = np.add(Total_Savings, facility_results[i].loc['TOTAL ENERGY SAVINGS SINCE BASELINE YEAR (MMBtu/Year)'].values)
        
        # Total improvement in energy intensity is weighted average using baseline year energy consumption
        Total_Improve += facility_results[i].loc['Total Improvement in Energy Intensity (%)']*facility_results[i].loc['TOTAL ACTUAL ENERGY (MMBtu)'].iloc[0]
    
    
    
    # ***** CALCULATE THE CORPORATE ROLL UP RESULTS ***** #
    
    n_util = len(Energy_Types)
    
    # Add energy consumption data
    EnPI_Results.iloc[0:n_util,:] = Actual_Energy
    EnPI_Results.loc['TOTAL PRIMARY ENERGY CONSUMED (MMBtu)'] = Actual_Energy.sum(axis=0)
    
    # Calculate baseline adjustment
    EnPI_Results.loc['Adjustment for Baseline Primary Energy Use (MMBtu/Yr)'] = Total_Savings + Actual_Energy.sum(axis=0) - Actual_Energy.sum(axis=0)[0]
    
    # Calculate adjusted baseline energy consumption
    EnPI_Results.loc['Adjusted Baseline Primary Energy Use (MMBtu/Yr)'] = Actual_Energy.sum(axis=0)[0] + EnPI_Results.loc['Adjustment for Baseline Primary Energy Use (MMBtu/Yr)'].values
    
    # Calculate the total improvement in energy intensity using the weighted sum of savings and baseline energy consumption
    EnPI_Results.loc['Total Improvement in Energy Intensity (%)'] = Total_Improve/EnPI_Results.loc['TOTAL PRIMARY ENERGY CONSUMED (MMBtu)'].iloc[0]
    
    # Calculate annual improvement from total improvement
    EnPI_Results.loc['Annual Improvement in Energy Intensity (%)'].iloc[0] = 0
    EnPI_Results.loc['Annual Improvement in Energy Intensity (%)'].iloc[1:] = EnPI_Results.loc['Total Improvement in Energy Intensity (%)'].iloc[1:].values - EnPI_Results.loc['Total Improvement in Energy Intensity (%)'].iloc[:-1].values
    
    # Add annual and total savings to the dataframe
    EnPI_Results.loc['New Energy Savings for Current Year (MMBtu/Year)'] = New_Savings
    EnPI_Results.loc['TOTAL ENERGY SAVINGS SINCE BASELINE YEAR (MMBtu/Year)'] = Total_Savings
    
    
    
    # ***** MAKE FINAL CHARTS AND PRINT THE DATAFRAME RESULTS ***** #
    
    # Print a graph showing the corporate progress over time
    plt.figure()
    gca = plt.gca()
    
    x = range(0,n_years)
    
    plt.plot(x, EnPI_Results.loc['Annual Improvement in Energy Intensity (%)'].values, marker='o', markersize=5, linewidth=2)
    plt.plot(x, EnPI_Results.loc['Total Improvement in Energy Intensity (%)'].values, marker='x', markersize=7.5, linewidth=2, markeredgewidth=2)
    
    gca.set_title('Corporate Energy Intensity Improvement')
    gca.set_xlabel('Reporting Year')
    gca.set_xticks(x)
    gca.set_xlim([-0.5, n_years-0.5])
    
    gca.set_ylabel('Percent Improvement')
    gca.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    
    plt.legend(['Annual Improvement in EI', 'Total Improvement in EI'],loc=2,fontsize=9)
    plt.grid()
    
    box = gca.get_position()
    gca.set_position([box.x0-0.02, box.y0, box.width, box.height])
    
    print('\nCorporate Roll Up Results:')
    
    # Print the table showing the corporate roll up results
    display(EnPI_Results)
        
    return EnPI_Results
