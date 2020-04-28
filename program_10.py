#!/bin/env python
# Created on March 25, 2020
#  by Keith Cherkauer
#
# This script servesa as the solution set for assignment-10 on descriptive
# statistics and environmental informatics.  See the assignment documention 
# and repository at:
# https://github.com/Environmental-Informatics/assignment-10.git for more
# details about the assignment.
#
'''
This script reads in two datasets (one for Wildcat Creek and one for Tippecanoe
River) and starts by removing gross error values. The datasets are then clipped
to the same date range and various statistics for the water discharge are calculated.
This includes mean, peak value, and 7-day low flow. This is done both on the basis
of a water year (10/01 to 9/30 the following year) and for each calendar month.
The results are then output into four files. Two CSV files contain the statistics
performed on each month/year. Two TXT files contain the average value of each of
these columns as tab delimited data.

@authors: tfield
@github: tfield156
'''
import pandas as pd
import scipy.stats as stats
import numpy as np

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # remove negative streamflow as gross error check
    DataDF['Discharge'].loc[(DataDF['Discharge'] < 0)] = np.NaN
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""

    # select only rows between the start and end dates (inclusive end points)
    DataDF = DataDF[startDate:endDate]
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""
    Qvalues.dropna() #remove NaN values
    qMean = Qvalues.mean() #calculate the mean
    totalDataPoints = Qvalues.count() #total number of data points
    greaterThanMean = Qvalues.loc[Qvalues > qMean].count() #Number of data points greater than the mean
    Tqmean = greaterThanMean/totalDataPoints #Ratio of points greater than the mean
    
    return ( Tqmean )

def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""
    Qvalues.dropna() #Remove NaN values
    totalDischarge = Qvalues.sum() #sum discharge for entire time period
    deltaDischarge = 0 #Initially no discharge changes
    for i in range(len(Qvalues)-1): #For the whole dataset
        deltaDischarge = deltaDischarge + np.abs(Qvalues[i]-Qvalues[i+1]) #Add the incremental change in discharge to the total change
    RBindex=deltaDischarge/totalDischarge #Ratio of incremental sum to total discharge
    
    return ( RBindex )

def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""
    
    Qvalues.dropna() #Remove NaN values
    moveAVG = Qvalues.rolling(window=7).mean() #7-day moving average
    val7Q = moveAVG.min() #minimum moving average value
    return ( val7Q )

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value. The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""
    
    qMedian = Qvalues.median() #get the median
    median3x = Qvalues.loc[Qvalues > (3*qMedian)].count() #count how many datapoints exceed it by more than 3x
    
    return ( median3x )

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    
    # calcluate water year annual statistics
    WYmean = DataDF['Discharge'].resample('AS-OCT',label='left').mean() #mean
    WYpeak = DataDF['Discharge'].resample('AS-OCT',label='left').max() #peak value
    WYmedian = DataDF['Discharge'].resample('AS-OCT',label='left').median() #median
    WYcoeffvar = DataDF['Discharge'].resample('AS-OCT',label='left').std()/WYmean *100 #coefficient of variance
    
    # initialize varaiables for repetitive computations
    WYskew = []
    WYtqmean = []
    WYrbindex = []
    WY7q = []
    WY3xmedian = []
    
    startDate = DataDF.index[0] #first data point
    endDate = startDate + pd.DateOffset(days=365) #a year later (leap dependent)
    if (endDate.month == 10): #if 10/01, shift back a day (not leap year)
        endDate = endDate - pd.DateOffset(days=1) # force last day of 9/30/XXXX
        
    for i in range(len(WYmean.index)): #For each water year
        yearDF = DataDF[startDate:endDate] #only the one year of data at a time
        
        # append the new annual value for each statistic
        WYskew.append(stats.skew(yearDF['Discharge'], nan_policy='omit')) #skew within water year
        WYtqmean.append(CalcTqmean(yearDF['Discharge'])) #TQ mean
        WYrbindex.append(CalcRBindex(yearDF['Discharge'])) #RB index
        WY7q.append(Calc7Q(yearDF['Discharge'])) #7 Day low flow
        WY3xmedian.append(CalcExceed3TimesMedian(yearDF['Discharge'])) #more than 3x the median days
    
        #get next water year date range
        startDate = endDate + pd.DateOffset(days=1) #10/01/XXXX
        endDate = startDate + pd.DateOffset(days=365) #a year later (leap dependent)
        if (endDate.month == 10): #if 10/01, shift back a day (not leap year)
            endDate = endDate - pd.DateOffset(days=1) # force last day of 9/30/XXXX

    #create dataframe out of all the statistics
    WYDF = pd.DataFrame({'Mean Flow':WYmean,'Peak Flow':WYpeak,'Median Flow':WYmedian,
                         'Coeff Var':WYcoeffvar,'Skew':WYskew,'Tqmean':WYtqmean,
                         'R-B Index':WYrbindex,'7Q':WY7q,'3xMedian':WY3xmedian})
    
    WYDataDF = WYDF #save to output variable
    return ( WYDataDF )

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""

    mMean = DataDF['Discharge'].resample('M',label='right').mean() #monthly mean
    mCV = DataDF['Discharge'].resample('M',label='right').std()/mMean *100 #coefficient of variance
    
    #shift month index to be first day of month, not last of previous month
    #Mean = mMean.shift(periods=1,freq='1D')
    #mCV = mCV.shift(periods=1,freq='1D')
    
    mTQ = []
    mRB = []
    
    startDate = DataDF.index[0] #first data point
    endDate = (startDate + pd.DateOffset(months=1)) - pd.DateOffset(days=1) #end of month
    
    for i in range(len(mMean.index)): #for each month
        monthDF = DataDF[startDate:endDate] #only the one month of data at a time
        
        mTQ.append(CalcTqmean(monthDF['Discharge'])) #TQ mean
        mRB.append(CalcRBindex(monthDF['Discharge'])) #RB index
        
        #get next month start and end dates
        startDate = endDate + pd.DateOffset(days=1)
        endDate = (startDate + pd.DateOffset(months=1)) - pd.DateOffset(days=1) #end of month
    
    
    mDF = pd.DataFrame({'Mean Flow':mMean,'Coeff Var':mCV,'Tqmean':mTQ,'R-B Index':mRB})
    MoDataDF=mDF.resample('MS',how='sum') #save to output variable
    
    return ( MoDataDF )

def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    
    # calculate the mean value for each of the annual statistics
    AnnualAverages = WYDataDF.mean()
    
    return( AnnualAverages )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    
    dates = MoDataDF.index #dates of each monthly statistic
    
    MonthAVG = []
    for i in range(12): #for each month
        MonthAVG.append(MoDataDF.loc[dates.month == (i+1)].mean()) #average of statistics for that month
        
    # create dataframe for all of the months
    MonthlyAverages = pd.DataFrame({1:MonthAVG[0],2:MonthAVG[1],
                                    3:MonthAVG[2],4:MonthAVG[3],
                                    5:MonthAVG[4],6:MonthAVG[5],
                                    7:MonthAVG[6],8:MonthAVG[7],
                                    9:MonthAVG[8],10:MonthAVG[9],
                                    11:MonthAVG[10],12:MonthAVG[11]})
    MonthlyAverages=MonthlyAverages.transpose()
    return( MonthlyAverages )

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define filenames as a dictionary
    # NOTE - you could include more than jsut the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    WYDataDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}
    
    # process input datasets
    for file in fileName.keys():
        
        print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # calculate descriptive statistics for each water year
        WYDataDF[file] = GetAnnualStatistics(DataDF[file])
        
        # calcualte the annual average for each stistic or metric
        AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
        
        print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])

        # calculate descriptive statistics for each month
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])
        
        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        
        print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])
    
    # append station info to each dataset and concatonate the two stations into a single dataframe
    anWildcat = WYDataDF['Wildcat']
    anWildcat['Station'] = 'Wildcat'
    anTippe = WYDataDF['Tippe']
    anTippe['Station'] = 'Tippe'
    annualMetrics = anWildcat.append(anTippe)
    
    monWildcat = MoDataDF['Wildcat']
    monWildcat['Station'] = 'Wildcat'
    monTippe = MoDataDF['Tippe']
    monTippe['Station'] = 'Tippe'
    monthlyMetrics = monWildcat.append(monTippe)
    
    anAVGwildcat = AnnualAverages['Wildcat'].to_frame().transpose()
    anAVGwildcat['Station'] = 'Wildcat'
    anAVGtippe = AnnualAverages['Tippe'].to_frame().transpose()
    anAVGtippe['Station'] = 'Tippe'
    annualAVGout = anAVGwildcat.append(anAVGtippe)

    monAVGwildcat = MonthlyAverages['Wildcat']
    monAVGwildcat['Station'] = 'Wildcat'
    monAVGtippe = MonthlyAverages['Tippe']
    monAVGtippe['Station'] = 'Tippe'
    monthlyAVGout = monAVGwildcat.append(monAVGtippe)
    
    #Output the four requested files in the required formats
    annualMetrics.to_csv('Annual_Metrics.csv',header=True)
    monthlyMetrics.to_csv('Monthly_Metrics.csv',header=True)
    annualAVGout.to_csv('Average_Annual_Metrics.txt',header=True,sep='\t',index=False)
    monthlyAVGout.to_csv('Average_Monthly_Metrics.txt',header=True,sep='\t')
    