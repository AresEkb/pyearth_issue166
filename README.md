A test script for https://github.com/scikit-learn-contrib/py-earth/issues/166

# Run with zero_tol=None

## Windows

```
Forward Pass
-------------------------------------------------------------------------
iter  parent  var  knot  mse           terms  gcv         rsq    grsq    
-------------------------------------------------------------------------
0     -       -    -     83406.187500  1      99260.256   0.000  0.000   
1     0       101  -1    34307.013889  2      68376.609   0.589  0.311   
2     0       36   -1    25185.570203  3      100742.281  0.698  -0.015  
3     2       39   -1    11316.205534  4      133023.151  0.864  -0.340  
4     2       21   -1    5061.939612   5      728919.304  0.939  -6.344  
5     0       42   -1    2690.722170   6      172206.219  0.968  -0.735  
6     1       72   -1    614.732328    7      5532.591    0.993  0.944   
7     2       108  -1    131.224915    8      447.252     0.998  0.995   
8     1       146  -1    1.381628      9      2.456       1.000  1.000   
-------------------------------------------------------------------------
Stopping Condition 1: Achieved RSQ value within threshold of 1

Pruning Pass
------------------------------------------------------
iter  bf  terms  mse       gcv         rsq    grsq    
------------------------------------------------------
0     -   9      2480.96   4410.593    0.970  0.956   
1     6   8      2480.96   8455.812    0.970  0.915   
2     7   7      2554.21   22987.898   0.969  0.768   
3     8   6      2690.72   172206.219  0.968  -0.735  
4     5   5      5061.94   728919.304  0.939  -6.344  
5     4   4      11316.21  133023.151  0.864  -0.340  
6     3   3      25185.57  100742.281  0.698  -0.015  
7     2   2      34307.01  68376.609   0.589  0.311   
8     1   1      83406.19  99260.256   0.000  0.000   
------------------------------------------------------
Selected iteration: 0

Earth Model
-------------------------------------------------------------------------------------------
Basis Function                                                       Pruned  Coefficient   
-------------------------------------------------------------------------------------------
(Intercept)                                                          No      7352.49       
ops_mail_type_year_min_lag_3                                         No      1.09535e-06   
delta_tc_t_cluster_lag_1                                             No      -7940.98      
delta_tc_t_cluster_lag_12*delta_tc_t_cluster_lag_1                   No      2983.59       
delta_min_rpo_lag_7*delta_tc_t_cluster_lag_1                         No      367.96        
delta_tc_t_cluster_lag_4                                             No      464.106       
ops_mail_type_month_mean_alltime_lag_1*ops_mail_type_year_min_lag_3  No      -0.00288506   
ops_rpo_month_mean_on_day_lag_1*delta_tc_t_cluster_lag_1             No      -1.77324      
trend_cluster_lag_11*ops_mail_type_year_min_lag_3                    No      -0.000154537  
-------------------------------------------------------------------------------------------
MSE: 2480.9588, GCV: 4410.5933, RSQ: 0.9703, GRSQ: 0.9556
[1334.11264062 1520.77594817 1839.5021724  1204.01597277 1815.69999402
 1392.89327199 1182.77221795  951.38210929  986.87059911 1158.89360342
 1084.02632246 1084.05514779  384.30440702]
```

## Linux

```
Forward Pass
-------------------------------------------------------------------------
iter  parent  var  knot  mse           terms  gcv         rsq    grsq    
-------------------------------------------------------------------------
0     -       -    -     83406.187500  1      99260.256   0.000  0.000   
1     0       101  -1    34307.013889  2      68376.609   0.589  0.311   
2     0       36   -1    25185.570203  3      100742.281  0.698  -0.015  
3     2       39   -1    11316.205534  4      133023.151  0.864  -0.340  
4     2       21   -1    5061.939612   5      728919.304  0.939  -6.344  
5     0       95   -1    2180.680815   6      139563.572  0.974  -0.406  
6     1       69   -1    458.951687    7      4130.565    0.994  0.958   
7     1       107  -1    16.335103     8      55.675      1.000  0.999   
-------------------------------------------------------------------------
Stopping Condition 1: Achieved RSQ value within threshold of 1

Pruning Pass
------------------------------------------------------
iter  bf  terms  mse       gcv         rsq    grsq    
------------------------------------------------------
0     -   8      3918.94   13356.853   0.953  0.865   
1     5   7      3918.94   35270.441   0.953  0.645   
2     7   6      3918.94   250812.027  0.953  -1.527  
3     6   5      5061.94   728919.304  0.939  -6.344  
4     4   4      11316.21  133023.151  0.864  -0.340  
5     3   3      25185.57  100742.281  0.698  -0.015  
6     2   2      34307.01  68376.609   0.589  0.311   
7     1   1      83406.19  99260.256   0.000  0.000   
------------------------------------------------------
Selected iteration: 0

Earth Model
---------------------------------------------------------------------------------
Basis Function                                             Pruned  Coefficient   
---------------------------------------------------------------------------------
(Intercept)                                                No      0.00194884    
ops_mail_type_year_min_lag_3                               No      -3.93648e-06  
delta_tc_t_cluster_lag_1                                   No      -7110.49      
delta_tc_t_cluster_lag_12*delta_tc_t_cluster_lag_1         No      2394.52       
delta_min_rpo_lag_7*delta_tc_t_cluster_lag_1               No      311.76        
ops_mail_type_year_max_lag_9                               No      3.74957       
ewma_lag_7*ops_mail_type_year_min_lag_3                    No      -0.000387031  
ops_mail_type_year_min_lag_9*ops_mail_type_year_min_lag_3  No      -0.00246829   
---------------------------------------------------------------------------------
MSE: 3918.9379, GCV: 13356.8535, RSQ: 0.9530, GRSQ: 0.8654
[1395.13384966 1592.58457915 1762.11176273 1217.62027763 1833.151756
 1306.3977743  1232.45238538 1009.97323641  942.47567301 1132.31554828
 1069.39074172 1061.39241573 -704.0142857 ]
```

# Run with zero_tol=1e-7

## Windows

```
Forward Pass
-------------------------------------------------------------------------
iter  parent  var  knot  mse           terms  gcv         rsq    grsq    
-------------------------------------------------------------------------
0     -       -    -     83406.187500  1      99260.256   0.000  0.000   
1     0       101  -1    34307.013889  2      68376.609   0.589  0.311   
2     0       36   -1    25185.570203  3      100742.281  0.698  -0.015  
3     2       39   -1    11316.205534  4      133023.151  0.864  -0.340  
4     2       21   -1    5061.939612   5      728919.304  0.939  -6.344  
5     0       42   -1    2690.722170   6      172206.219  0.968  -0.735  
6     1       39   -1    792.198595    7      7129.787    0.991  0.928   
7     0       94   -1    122.858350    8      418.736     0.999  0.996   
8     1       12   -1    30.043092     9      53.410      1.000  0.999   
-------------------------------------------------------------------------
Stopping Condition 1: Achieved RSQ value within threshold of 1

Pruning Pass
--------------------------------------------------------
iter  bf  terms  mse       gcv          rsq    grsq     
--------------------------------------------------------
0     -   9      30.04     53.410       1.000  0.999    
1     8   8      122.86    418.736      0.999  0.996    
2     3   7      762.98    6866.842     0.991  0.931    
3     7   6      1520.72   97326.248    0.982  0.019    
4     5   5      9348.34   1346160.941  0.888  -12.562  
5     4   4      14868.68  174782.828   0.822  -0.761   
6     6   3      25185.57  100742.281   0.698  -0.015   
7     2   2      34307.01  68376.609    0.589  0.311    
8     1   1      83406.19  99260.256    0.000  0.000    
--------------------------------------------------------
Selected iteration: 0

Earth Model
-----------------------------------------------------------------------------
Basis Function                                          Pruned  Coefficient  
-----------------------------------------------------------------------------
(Intercept)                                             No      5296.12      
ops_mail_type_year_min_lag_3                            No      -5.20867     
delta_tc_t_cluster_lag_1                                No      -5403.65     
delta_tc_t_cluster_lag_12*delta_tc_t_cluster_lag_1      No      711.633      
delta_min_rpo_lag_7*delta_tc_t_cluster_lag_1            No      546.462      
delta_tc_t_cluster_lag_4                                No      770.867      
delta_tc_t_cluster_lag_12*ops_mail_type_year_min_lag_3  No      2.44409      
ops_mail_type_year_max_lag_8                            No      0.706195     
delta_min_rpo_lag_1*ops_mail_type_year_min_lag_3        No      -0.063301    
-----------------------------------------------------------------------------
MSE: 30.0431, GCV: 53.4099, RSQ: 0.9996, GRSQ: 0.9995
[1393.84495835 1476.44133394 1918.35827176 1222.48876063 1762.90876033
 1332.95791498 1247.86904533 1003.85476839  953.47734817 1121.96673427
 1061.83210384 1059.          341.54427033]
```

## Linux

```
Forward Pass
-------------------------------------------------------------------------
iter  parent  var  knot  mse           terms  gcv         rsq    grsq    
-------------------------------------------------------------------------
0     -       -    -     83406.187500  1      99260.256   0.000  0.000   
1     0       101  -1    34307.013889  2      68376.609   0.589  0.311   
2     0       36   -1    25185.570203  3      100742.281  0.698  -0.015  
3     2       39   -1    11316.205534  4      133023.151  0.864  -0.340  
4     2       21   -1    5061.939612   5      728919.304  0.939  -6.344  
5     0       42   -1    2690.722170   6      172206.219  0.968  -0.735  
6     1       39   -1    792.198595    7      7129.787    0.991  0.928   
7     0       94   -1    122.858350    8      418.736     0.999  0.996   
8     1       12   -1    30.043092     9      53.410      1.000  0.999   
-------------------------------------------------------------------------
Stopping Condition 1: Achieved RSQ value within threshold of 1

Pruning Pass
--------------------------------------------------------
iter  bf  terms  mse       gcv          rsq    grsq     
--------------------------------------------------------
0     -   9      30.04     53.410       1.000  0.999    
1     8   8      122.86    418.736      0.999  0.996    
2     3   7      762.98    6866.842     0.991  0.931    
3     7   6      1520.72   97326.248    0.982  0.019    
4     5   5      9348.34   1346160.941  0.888  -12.562  
5     4   4      14868.68  174782.828   0.822  -0.761   
6     6   3      25185.57  100742.281   0.698  -0.015   
7     2   2      34307.01  68376.609    0.589  0.311    
8     1   1      83406.19  99260.256    0.000  0.000    
--------------------------------------------------------
Selected iteration: 0

Earth Model
-----------------------------------------------------------------------------
Basis Function                                          Pruned  Coefficient  
-----------------------------------------------------------------------------
(Intercept)                                             No      5296.12      
ops_mail_type_year_min_lag_3                            No      -5.20867     
delta_tc_t_cluster_lag_1                                No      -5403.65     
delta_tc_t_cluster_lag_12*delta_tc_t_cluster_lag_1      No      711.633      
delta_min_rpo_lag_7*delta_tc_t_cluster_lag_1            No      546.462      
delta_tc_t_cluster_lag_4                                No      770.867      
delta_tc_t_cluster_lag_12*ops_mail_type_year_min_lag_3  No      2.44409      
ops_mail_type_year_max_lag_8                            No      0.706195     
delta_min_rpo_lag_1*ops_mail_type_year_min_lag_3        No      -0.063301    
-----------------------------------------------------------------------------
MSE: 30.0431, GCV: 53.4099, RSQ: 0.9996, GRSQ: 0.9995
[1393.84495835 1476.44133394 1918.35827176 1222.48876063 1762.90876033
 1332.95791498 1247.86904533 1003.85476839  953.47734817 1121.96673427
 1061.83210384 1059.          341.54427033]
```
