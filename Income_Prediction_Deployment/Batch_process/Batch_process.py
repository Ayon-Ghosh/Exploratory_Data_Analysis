# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import os
import json
from flask import Flask, render_template, request,jsonify
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import csv


class batch_data_prep:
    def __init__(self ,dataframe, threshold):
        self.dataframe = dataframe
        self.threshold = threshold

    def categorical_feature(self, dataframe, threshold):
        categorical_features = []
        for col in dataframe.columns:
            if len(dataframe[col].unique() ) <= threshold or dataframe[col].dtypes == 'object':
                # print(f'{col}:{dataframe[col].unique()}: {dataframe[col].dtypes}')
                categorical_features.append(col)
            else:
                continue
        return categorical_features


    def data_visualization(self, dataframe ,categorical_feature_list ,cols=2, width=20, height=45, hspace=0.8, wspace=0.8):
        # Use matplotlib style settings from a style specification.
        plt.style.use('fivethirtyeight')
        # Create a new figure
        fig = plt.figure(figsize=(width, height))
        # customizing the subplots
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
        rows = math.ceil(float(dataframe.shape[1]) / cols)
        # iterating over the columns and then showing the data distribution in various columns

        for i, column in enumerate(dataframe.columns):

            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_title(column)
            if column not in categorical_feature_list:
                # with out using the following parameters:
                # hist = False, rug = True, rug_kws = {'color' : 'r'}, kde_kws = {'bw' : 1}
                # we will run into error: RuntimeError: Selected KDE bandwidth is 0. Cannot estimate density
                # refer link: https://stackoverflow.com/questions/60596102/seaborn-selected-kde-bandwidth-is-0-cannot-estimate-density
                g = sns.distplot(dataframe[column] ,hist = False, rug = True, rug_kws = {'color' : 'r'}, kde_kws = {'bw' : 1})

                plt.xlabel(column ,fontsize=12)
                plt.xticks(rotation=25)
            else:
                g = sns.countplot(y=column, data=dataframe)
                if column == 'education' or column == 'occupation' or column == 'education_num' or column == 'native_country':
                    plt.yticks(rotation=0)
                else:
                    plt.yticks(rotation=25)
        fig.savefig('static/full_figure_1.jpg')


    def encoder(self ,dataframe):
        df = dataframe.copy()
        for column in df.columns:
            if df[column].dtypes == 'object':
                length_of_unique_value_set = list(range(df[column].nunique()))
                value_set = list(df[column].unique())
                temp_mapping_dict = dict(zip(value_set ,length_of_unique_value_set))
                df[column] = df[column].map(temp_mapping_dict)
        return df

    def data_visualization_groupby_target(self ,dataframe ,categorical_feature_list ,cols=5, width=20, height=30, hspace=0.2, wspace=0.5):
        # Use matplotlib style settings from a style specification.
        plt.style.use('fivethirtyeight')
        # Create a new figure
        fig = plt.figure(figsize=(width ,height))
        # customizing the subplots
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
        rows = math.ceil(float(len(categorical_feature_list)) / cols)
        # iterating over the columns and then showing the data distribution in various columns

        for i, column in enumerate(categorical_feature_list):

            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_title(column)
            # temp_df = dataframe.groupby([column,'wage_class'])['wage_class'].count()
            g = sns.countplot(column ,data=dataframe ,hue='wage_class')
            plt.xlabel(column ,fontsize=12)
            plt.xticks(rotation=90)
        fig.savefig('static/full_figure_2.jpg')