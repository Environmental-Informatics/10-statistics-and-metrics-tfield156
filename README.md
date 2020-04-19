# Environmental Informatics

## Assignment 10 - Descriptive Statistics and Environmental Metrics

### Lab Objectives

On completion of this lab, students will be able to:

1. Calculate descriptive statistics for a dataset;
3. Calculate streamflow metrics further describing the distribution of daily streamflow and that are correlated with aquatic ecosystem health; and
3. Generate summary tables that put the streamflow data into context using statistics and metrics.

### Reading Assignment

For descriptive statistics, you can start with any basic statistics textbook, as they will all discuss the concepts of measuring the central tendency and variability of a dataset.  For a brief review of the concepts in the context of data science (or informatics), check out:

- [Introduction to descriptive statistics](https://towardsdatascience.com/intro-to-descriptive-statistics-252e9c464ac9)

For environmental metrics, please read these two papers by Derek Booth and two of his students (posted to Blackboard under Reading Materials -> Analysis Metrics).  Both seek to identify environmental metrics with predictive ability for water quality.  The first uses metrics to quantify the most ecologically significant changes in daily streamflow characteristics, while the second quantifies observed differences in land use in the upstream area near sampling sites.

- Konrad, C.P. and Booth, D.B. (2005) Hydrologic Changes in Urban Streams and Their Ecological Significance. American Fisheries Society Symposium, 47, 157-177.

- McBride, M. and Booth, D.B. (2005), URBAN IMPACTS ON PHYSICAL STREAM CONDITION: EFFECTS OF SPATIAL SCALE, CONNECTIVITY, AND LONGITUDINAL TRENDS1. JAWRA Journal of the American Water Resources Association, 41: 565-580. [doi:10.1111/j.1752-1688.2005.tb03755.x](doi:10.1111/j.1752-1688.2005.tb03755.x)

### The Lab Assignment

This week’s assignment is to calculate basic descriptive statistics and environmental metrics for two rivers with similar climate and drainage areas.  Do this by reading the two files into your Python code, checking for gross errors, clipping the two datasets to a consistent time period, and calculating the prescribed statistics and metrics.  

1. Use the files **TippecanoeRiver_Discharge_03331500_19431001-20200315.txt** and **WildcatCreek_Discharge_03335000_19540601-20200315.txt** provided with this repository.

   - These files contain daily discharge in cubic feet per second (cfs) for a single USGS gaging station.
   - Data file is white space delimited.
   - Column order and units are provided in the file header.
     
2. Modify the Python script template called program_10_template.py to complete this assignment.  The template contains code defining function names and input/output parameters, as well as comment text describing what each function should accept as parameters, return as variables, and functionally what each function should do.  **DO NOT** change the name or order of the parameters being sent to or from each function.  The template defines functions that the autograder program is going to import and evaluate.  Changing the program name or the function definitions, will cause the autograder to fail even if your code "works" for you.  

2. Copy the Python script template to a new script called **program_10.py** in the same folder, and edit this script so that it does the following:

   - Imports both streamflow file as DataFrames, using date as the index.  
     - Use two dataframes, or a list of dataframes so that the dataframe containing streamflow from each river can be processed by each of the following functions.
     - Make sure that this function removes invalid streamflow data values.  The USGS file typically skips days with no measurement, but there are times when non-number values are used to fill for the discharge measurement when a final decision on its availability or quality has not been made.
     - Modify the provided function `ReadData()` to remove negative streamflow values as a gross error check.
     - Report only the total number of days missing observations at the end of the function, you do not need to count those failing the gross error check separately.
   - Complete the function `ClipData` to clip the data to the date range: October 1, 1969 through September 30, 2019.  This will provide you with 50 water years of streamflow data for your analysis.  The USGS defines a water year as starting on October 1 of any year, for example the water year 2010 started on October 1, 2010 and ran through September 30, 2011.
   - Complete the function `GetAnnualStatistics` to build a new dataframe to contain annual (water year) values for the following metrics:
     
     | Full Name | DF Name | Description |
     |-----|-----|-----|
     | Annual Mean Flow | `Mean Flow` | Average annual streamflow for a given year. Can apply standard Pandas function. |
     | Annual Peak Flow | `Peak Flow` | Maximum streamflow value in a given year. Can apply standard Pandas function. |
     | Annual Median Flow | `Median Flow` |  Median streamflow value in a given year. Can apply standard Pandas function. |
     | Coefficient of Variation | `Coeff Var` | Coefficient of variation of streamflow in a given year. Calculated as the standard deviation divided by the mean annual streamflow multiplied by 100. |
     | Skewness | `Skew` | The skewness of the dataset.  Can make use of the SciPy Stats skew function. |
     | T-Qmean | `Tqmean` | The fraction of time (days) that streamflow exceeds mean annual streamflow (Qmean).  Complete the function `CalcTqmean` to accept a Series of daily streamflow values, calculate the mean flow, and return the number of days on which daily streamflow is greater than the mean flow. |
     | Richards-Baker Flashiness Index | `R-B Index` | This is a measure of how much streamflow can change in a day ('flashiness').  Complete the function `CalcRBindex` to accept a Series of daily streamflow values, and return the R-B Index value.  The R-B Index is calculated by dividing the sum of the absolute values of day-to-day streamflow change by total discharge volumes for each year. |
     | 7-day Low Flow | `7Q` | The 7-day low flow is a common measurement of low flow in stream discharge, as metrics like minimum daily flow can fail as no flow or flow that is too low to measure is not unexpected in some cases.  Complete the function `Calc7Q` to accept a Series of Daily streamflow values, and return the 7-day low flow value. The index is calculated by computing a 7-day moving average for the annual dataset, and picking the lowest average flow in any 7-day period. |
     | Flow Exceeding 3 Times Median Flow | `3xMedian` | This is a measure of the frequency of high flow (but not necessarily the highest flow) events.  Complete the function `CalcExceed3TimesMedian` to accept a Series of daily streamflow values, and return the number of days that streamflow exceeds 3 times the median flow of the same period. |
     
     - Use the `Table Name` as the column header in your Pandas DataFrame.
     - Look into using the Pandas [`resample` method](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling) to aggregate your daily time series into annual values.  You will want to review *Anchored Offsets* for appropriate ways to sample data and maintain the Water Year using October 1 as the anchor for resultant dates. Pandas functions such as `mean` can be applied directly during the resample process, but to apply your custom built functions look into the Pandas `apply` method. 
     
   - Complete the function `GetMonthlyStatistics` to build a new dataframe to contain monthly values for the following descriptive statistics and metrics:
   
     | Full Name | DF Name | Description |
     |-----|-----|-----|
     | Annual Mean Flow | `Mean Flow` | Average annual streamflow for a given year. Can apply standard Pandas function. |
     | Coefficient of Variation | `Coeff Var` | Coefficient of variation of streamflow in a given year. Calculated as the standard deviation divided by the mean annual streamflow multiplied by 100. |
     | T-Qmean | `Tqmean` | The fraction of time (days) that streamflow exceeds mean annual streamflow (Qmean).  Complete the function `CalcTqmean` to accept a Series of daily streamflow values, calculate the mean flow, and return the number of days on which daily streamflow is greater than the mean flow. |
     | Richards-Baker Flashiness Index | `R-B Index` | This is a measure of how much streamflow can change in a day ('flashiness').  Complete the function `CalcRBindex` to accept a Series of daily streamflow values, and return the R-B Index value.  The R-B Index is calculated by dividing the sum of the absolute values of day-to-day streamflow change by total discharge volumes for each year. |
    
     -  This function should be very similar to the annual version, but reporting for months.  There should be no reason to rewrite any of the metric calculation functions, if you wrote them to work with *any* length data series provided.
     
   - Complete the function `GetAnnualAverages` and `GetMonthlyAverages` to compute annual average values for all metrics.  
     - Annual average function should return a Series with a single average value for each statistic of metric.
     - Monthly average function should return a DataFrame with 12 monthly values for each metric.
     
3. Output the annual and monthly metrics tables to CSV files in the current directory called **Annual_Metrics.csv** and **Monthly_Metrics.csv**, respectively.  Each file should contain data for both rivers, with one column being the short river name "Tippe" or "Wildcat", with the column name "Station".

4. Output the annual and monthly averages to TAB delimited files in the current directory called **Average_Annual_Metrics.txt** and **Average_Monthly_Metrics.txt**, respectively.  Each file should contain data for both rivers, with one column being the short river name "Tippe" or "Wildcat", with the column name "Station".

5. Be sure that the script has a complete header comment block, appropriate in-line comments, and runs without intervention relative to where the datafile is stored in the repository.

#### What to turn in...

The following should be included in your GitHub repository:

1. A working program called **program_10.py**, which conforms to the template provided with the original repository.

2. The original data files, **TippecanoeRiver_Discharge_03331500_19431001-20200315.txt** and **WildcatCreek_Discharge_03335000_19540601-20200315.txt**, provided with the repository.

3. The four output files from the program: **Annual_Metrics.csv**, **Monthly_Metrics.csv**, **Average_Annual_Metrics.txt** and **Average_Monthly_Metrics.txt**.

4. Put your input file, output files, and processing script in the assignment repository and push to GitHub. 

5. The instructor will follow-up with an invitation to submit the assignment through GradeScope.  Follow the provided instructions to submit the assignment using GitHub.  The AutoGrader will return a report on whether the submitted code met the performance checks.  If errors are found, fix the code, and resubmit.

#### Grading Rubric (50 pts Total)

| Question | Description | Score |
| -------- | ----------- | ----- |
| 1. | Write a Python script to complete the following analysis | (35.0 pts) |
| 1.1. | - Import the entire file in as a DataFrame, using date as the index | 3.0 pts |
| 1.2. | - Clip data to correct analysis time period. | 3.0 pts |
| 1.3. | - Build Water Year metrics table | 4.0 pts |
| 1.4. | - Build Monthly metrics table | 4.0 pts |
| 1.5. | - Calculate annual metrics (1 pt per metric). | 9.0 pts |
| 1.6. | - Calculate monthly metrics (1 pt per metric). | 4.0 pts |
| 1.7. | - Calculate annual average metrics.  | 4.0 pts |
| 1.8. | - Calculate annual monthly average metrics. | 4.0 pts |
| 2. | Program should have clear and concise header and in-line comments | 5 pts |
| 3. | Output files: | (10 pts) |
| 3.1. | - Annual_Metrics.csv file contains all annual metrics and is CSV format. | 2.5 pts |
| 3.2. | - Monthly_Metrics.csv file contains all monthly metrics and is CSV format. | 2.5 pts |
| 3.3. | - Average_Annual_Metrics.txt file contains annual average values in a TAB delimited format. | 2.5 pts |
| 3.4. | - Average_Monthly_Metrics.txt file contains annual monthly average values in a TAB delimited format. | 2.5 pts |

#### General output file the program

The following is example output from the instructors version of the code, which uses the print statements embedded in the template.  Use it as a guide for evaluating you program's performance.

```
 ================================================== 
  Working on Wildcat 
 ================================================== 

-------------------------------------------------- 

Raw data for Wildcat...

          site_no     Discharge
count    24030.0  23967.000000
mean   3335000.0    821.680131
std          0.0   1350.045050
min    3335000.0     46.000000
25%    3335000.0    183.000000
50%    3335000.0    382.000000
75%    3335000.0    840.000000
max    3335000.0  22100.000000 

Missing values: 63


-------------------------------------------------- 

Selected period data for Wildcat...

          site_no     Discharge
count    18262.0  18202.000000
mean   3335000.0    865.880656
std          0.0   1373.141800
min    3335000.0     53.000000
25%    3335000.0    198.000000
50%    3335000.0    414.000000
75%    3335000.0    908.000000
max    3335000.0  21000.000000 

Missing values: 60


-------------------------------------------------- 

Summary of water year metrics...

          site_no    Mean Flow     Peak Flow  ...  R-B Index          7Q    3xMedian
count       50.0    50.000000     50.000000  ...  49.000000   50.000000   50.000000
mean   3335000.0   866.472457  10387.400000  ...   0.271422  107.305714   63.360000
std          0.0   261.492725   4337.589291  ...   0.032684   30.509673   19.723352
min    3335000.0   263.553279   1750.000000  ...   0.167063   57.714286   23.000000
25%    3335000.0   714.452186   7407.500000  ...   0.254813   83.785714   49.250000
50%    3335000.0   866.426402  10055.000000  ...   0.272662   99.192857   60.500000
75%    3335000.0  1020.976612  12700.000000  ...   0.289134  130.785714   75.000000
max    3335000.0  1460.142466  21000.000000  ...   0.338636  200.428571  120.000000

[8 rows x 10 columns] 

Annual water year averages...

 site_no      3.335000e+06
Mean Flow    8.664725e+02
Peak Flow    1.038740e+04
Median       4.290800e+02
Coeff Var    1.488176e+02
Skew         3.979348e+00
TQmean       2.626813e-01
R-B Index    2.714219e-01
7Q           1.073057e+02
3xMedian     6.336000e+01
dtype: float64
-------------------------------------------------- 

Summary of monthly metrics...

          site_no    Mean Flow  Coeff Variation      TQmean   R-B Index
count      600.0   599.000000       599.000000  600.000000  597.000000
mean   3335000.0   872.357368        65.511054    0.338544    0.216880
std          0.0   824.597297        37.021790    0.088007    0.113318
min    3335000.0    61.645161         3.177400    0.000000    0.015119
25%    3335000.0   257.437212        35.847740    0.266667    0.122772
50%    3335000.0   588.607143        61.917185    0.333333    0.215508
75%    3335000.0  1225.897849        87.298404    0.400000    0.291989
max    3335000.0  4945.566667       213.572436    0.645161    0.663586 

Annual Monthly Averages...

       site_no    Mean Flow  Coeff Variation    TQmean  R-B Index
Date                                                            
1     3335000  1045.404774        77.407106  0.334839   0.224337
2     3335000  1147.643522        73.765263  0.341995   0.225558
3     3335000  1422.225161        65.809690  0.361935   0.221199
4     3335000  1373.372424        65.241920  0.354000   0.215812
5     3335000  1136.836076        64.278213  0.365806   0.216184
6     3335000  1073.021333        75.416638  0.324000   0.287210
7     3335000   663.146065        66.078786  0.336129   0.247715
8     3335000   331.752194        70.048690  0.296129   0.243332
9     3335000   345.425400        51.547777  0.316000   0.179261
10    3335000   371.321677        46.285064  0.351613   0.150802
11    3335000   640.754200        65.009951  0.322667   0.193044
12    3335000   922.675161        65.218896  0.357419   0.199482

 ================================================== 
  Working on Tippe 
 ================================================== 

-------------------------------------------------- 

Raw data for Tippe...

          site_no     Discharge
count    27926.0  27926.000000
mean   3331500.0    897.967020
std          0.0    834.547461
min    3331500.0     87.000000
25%    3331500.0    343.000000
50%    3331500.0    630.000000
75%    3331500.0   1170.000000
max    3331500.0   9410.000000 

Missing values: 0


-------------------------------------------------- 

Selected period data for Tippe...

          site_no     Discharge
count    18262.0  18262.000000
mean   3331500.0    955.697404
std          0.0    875.525342
min    3331500.0    103.000000
25%    3331500.0    370.000000
50%    3331500.0    671.000000
75%    3331500.0   1250.000000
max    3331500.0   9410.000000 

Missing values: 0


-------------------------------------------------- 

Summary of water year metrics...

          site_no    Mean Flow    Peak Flow  ...  R-B Index          7Q    3xMedian
count       50.0    50.000000    50.000000  ...  50.000000   50.000000   50.000000
mean   3331500.0   955.770639  4896.000000  ...   0.090484  220.091429   35.140000
std          0.0   243.392988  2009.161669  ...   0.011515   58.537925   26.470631
min    3331500.0   509.909589  1600.000000  ...   0.071168  106.285714    0.000000
25%    3331500.0   831.069672  3215.000000  ...   0.082771  176.285714   12.250000
50%    3331500.0   922.210959  4680.000000  ...   0.091693  205.285714   28.500000
75%    3331500.0  1135.960274  6190.000000  ...   0.096382  246.857143   57.000000
max    3331500.0  1439.430137  9410.000000  ...   0.118551  353.000000  104.000000

[8 rows x 10 columns] 

Annual water year averages...

 site_no      3.331500e+06
Mean Flow    9.557706e+02
Peak Flow    4.896000e+03
Median       7.065100e+02
Coeff Var    8.296229e+01
Skew         1.940724e+00
TQmean       3.611404e-01
R-B Index    9.048366e-02
7Q           2.200914e+02
3xMedian     3.514000e+01
dtype: float64
-------------------------------------------------- 

Summary of monthly metrics...

          site_no    Mean Flow  Coeff Variation      TQmean   R-B Index
count      600.0   600.000000       600.000000  600.000000  600.000000
mean   3331500.0   957.646535        32.930403    0.408629    0.078044
std          0.0   689.762885        19.693812    0.091182    0.038389
min    3331500.0   140.903226         2.852413    0.129032    0.007744
25%    3331500.0   422.274194        18.960021    0.354839    0.049303
50%    3331500.0   755.616667        28.235982    0.419355    0.073976
75%    3331500.0  1330.080645        42.070187    0.466667    0.100728
max    3331500.0  4239.032258       132.525184    0.666667    0.226103 

Annual Monthly Averages...

       site_no    Mean Flow  Coeff Variation    TQmean  R-B Index
Date                                                            
1     3331500  1118.534194        37.602774  0.384516   0.086071
2     3331500  1243.437241        43.295689  0.392217   0.094091
3     3331500  1586.004516        33.232188  0.425161   0.077306
4     3331500  1555.052000        33.279283  0.450000   0.077291
5     3331500  1219.602581        35.547500  0.402581   0.081068
6     3331500  1093.350000        31.983243  0.444000   0.087395
7     3331500   680.346452        36.537358  0.410968   0.089949
8     3331500   471.619355        31.021821  0.407097   0.073490
9     3331500   409.196667        26.538521  0.376667   0.066408
10    3331500   491.221290        24.748246  0.412903   0.056137
11    3331500   688.158000        30.050560  0.400667   0.072524
12    3331500   935.236129        31.327649  0.396774   0.074801
```
