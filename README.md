# Stock Market Price Prediction Using RNN

## Overview

This repository contains code for predicting stock market prices using Recurrent Neural Networks (RNN). The model is trained on historical stock data and can be used to make predictions for future stock prices.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [License](#license)

## Introduction

Stock_Market_Price_Prediction_Using-RNN is a project aimed at leveraging deep learning techniques, specifically Recurrent Neural Networks, to predict stock market prices. The model is trained on historical stock data, and the predictions can be utilized for decision-making in financial markets.

## Features

- **Data Visualization:** Visualize historical stock data, descriptive statistics, and trends over time.
- **RNN Model:** Implement a Recurrent Neural Network for stock price prediction.
- **Next Day Prediction:** Provide predictions for the next day's stock price based on the trained model.

## Requirements

- Python 3.10.6
- Libraries: yfinance, pandas, numpy, matplotlib, keras, streamlit

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/Stock_Market_Price_Prediction_Using-RNN.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Navigate to the project directory:

    ```bash
    cd Stock_Market_Price_Prediction_Using-RNN
    ```

2. Run the Streamlit app:

    ```bash
    streamlit run Stock_prediction.py
    ```

3. Open the provided URL in your web browser to interact with the Stock Trend Prediction interface.

## File Structure

- **app.py:** Streamlit application for user interaction.
- **stock_prediction.h5:** Pre-trained RNN model for stock price prediction.
- **other_files_and_folders:** Additional files and folders as needed.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to contribute, open issues, or provide feedback. Happy predicting!

