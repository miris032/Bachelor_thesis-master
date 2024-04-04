This project is basis on slidSHAPs project in bit.ly/3CRBlH2. The following python 
files are some mainly new additions for this thesis. 

1. run_biobank.py: to get the slidSHAP series plots, time series plots, 
and detected drifts for biobank data. It calls 
draw_biobank_timeseries.py, draw_slidSHAPs_plot_biobank.py, to get these plots. 

2. run_stock.py: to get the slidSHAP series plots, time series plots, 
and detected drifts for stock financial data. It calls draw_stock_timeseries.py, 
draw_slidSHAPs_plot_stock.py to get these plots. 

3. events_insertion.py: used to insert synthetic drift type 1 and type 2. 
It save e.g., Open_hourly as Open_hourly(events2).csv after inserting drift type 1, 
and Open_hourly(events).csv after inserting drift type 2. 

4. evaluation.py: use the main method in evaluation.py to print the results of 
Accuracy, Precision, Recall, F1-Score, and avgDELAY for all datasets with synthetic drifts insertion. 