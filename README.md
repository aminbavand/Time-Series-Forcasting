# Time-Series-Forcasting

This repository contains the codes for time series forcasting of Mackey-Glass and Santa Fe Laser datasets. Feedforward neural network and LSTM have been used and
the results have been compared. A complete report regarding the project is provided in README.pdf file.

## Getting Started

To run the project please do the following:

1. Clone the repository
```
$ git clone git@github.com:aminbavand/Time-Series-Forcasting.git
```

2. Check into the cloned repository
```
$ cd Time-Series-Forcasting
```

3. Run main.py file
```
$ python3 main.py
```

Note that lines 16, 17 in the main.py file determine the algorithm and the dataset as follows:
```
 (alg=1 =>LSTM), (alg=2 => FeedForward)
 (dataset=1 =>Mackey-Glass), (dataset=2 => Santa Fe Laser)
```
You can change these two lines to switch between the algorithms and datasets as you desire.
