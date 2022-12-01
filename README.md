# Stock Price prediction using Deep Learning Models: LSTM, Simple RNN, GRU

we propose the accurate prediction on stock market data gathered from 2017–2022 by implementing a basic Recurrent Neural Network, LSTM, and GRU machine learning models. High Price, Opening Price, Closing Price, and Low Price are the four primary indicators in this stock data set. We will perform the same experiment using data from Amazon, Apple, Google, Oracle, Microsoft, and Tesla, and find out the best model that yields the best predicted results in every case. 

## Dataset description -

We downloaded our dataset from - https://www.tiingo.com/

The dataset contains data attributes named: 

Name          | Description\
-------------------------------------------------------------------------\
Symbol        | Name code of the company\
Date          | specifies the trading date \
Open          | opening price \
High          | maximum price during the day \
Low           | minimum price during the day \
Close         | close price adjusted for splits \
Adj Close     | adjusted close price adjusted for both  dividends and splits \
Adj Open      | adjusted high price adjusted for both  dividends and splits\
Adj Low       | adjusted low price adjusted for both  dividends and splits\
Adj High      | adjusted high price adjusted for both dividends and splits\
Volume        | the number of shares that changed hands during a given day\
Adj Volume    | adjusted volume price adjusted for both dividends and splits\
Div Cash      | distribution of funds or money paid to stockholders\
Split Factor  | Ratio in which additional shares are issued to shareholders

![Dataset description](/Users/jagrutidhondage/Desktop)

We will perform the same experiment using data from Amazon, Apple, Google, Oracle, Microsoft, and Tesla
For each company the dataset can be downloaded as -

    data = pr.get_data_tiingo(company, api_key="0f6351ae343427e511f4d085681db7e303ffb969")
    data.to_csv(company+'.csv')
    data = pd.read_csv(company+'.csv')

We divided the dataset in 80:20 ratio for training and testing

### Environment setup

We developed this project on Jupyter notebooks. To run the code you can either use google colab or jupyter notebooks. Below are the steps to setup the code on Jupyter notebooks.

Step 1 - Install python3\
Step 2 - \
    python3 --version\
    pip3 --version\
Step 3 - \
    pip3 install --upgrade pip\
Step 4 -\
    pip3 install jupyter
    
After successfully installing jupyter notebooks on your system, open it.\
Step 5 - \
    jupyter notebook
    
### How to run code


We have following four files -

1. Models
2. Companies
3. Prices
4. Time_Step_Values

Models.ipynb contains 3 models that we worked on - LTSM, RNN, GRU
Run this code to get predictions and comparisons of these models on the Oracle dataset

Companies.ipynb contains evaluation of LTSM model on 6 companies namely Amazon, Apple, Google, Oracle, Microsoft and Tesla
Run this code to get actual and predicted values along with graphs visualizing mean squared errors for each company

Prices.ipynb contains evaluation of LTSM model on Oracle dataset based on price types namely Open, Close, High and Low values
Run this code to get actual and predicted values along with graphs visualizing mean squared errors for each varying input value

Time_Step_Values.ipynb contains evaluation of LTSM model on Oracle dataset based on varying time steps of 30, 40, 50, 70, 90
Run this code to get actual and predicted values of all time steps along with graphs visualizing the mean squared errors
