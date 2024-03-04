#! /usr/bin/env python3

"""
Sherlock: Find Usernames Across Social Networks Module

This module contains the main logic to search for usernames at social
networks.
"""

import csv
import signal
import os
import platform
import sys
import requests
import threading
import json
import pandas as pd
import matplotlib.pyplot as plt
import sherlock
import gensim
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from time import monotonic
from colorama import init
from argparse import ArgumentTypeError

module_name = "Sherlock: Find Usernames Across Social Networks"
__version__ = "0.14.3"


class QueryStatus:
    CLAIMED = "Claimed"
    AVAILABLE = "Available"
    UNKNOWN = "Unknown"


class QueryResult:
    def __init__(self, username, site_name, site_url_user, status, query_time=None, context=None):
        self.username = username
        self.site_name = site_name
        self.site_url_user = site_url_user
        self.status = status
        self.query_time = query_time
        self.context = context

    def __str__(self):
        return f"Username: {self.username}, Site: {self.site_name}, URL: {self.site_url_user}, Status: {self.status}, Query Time: {self.query_time}, Context: {self.context}"


class QueryNotifyPrint:
    def __init__(self, verbose=0):
        self.verbose = verbose

    def start(self, username):
        if self.verbose:
            print(f"Searching for username: {username}")

    def update(self, result):
        if self.verbose:
            print(result)


def facebook_advanced_search(account_id, verbose=0):
    url = f"https://graph.facebook.com/{account_id}"
    params = {
        'fields': 'id,name,email,first_name,last_name,gender,birthday,location,work,education,relationship_status',
        'access_token': 'your_access_token_here'  # Replace with your Facebook Graph API access token
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if 'error' in data:
            print(f"Error: {data['error']['message']}")
        else:
            print("Facebook Account Information:")
            print(f"Name: {data.get('name')}")
            print(f"ID: {data.get('id')}")
            print(f"Email: {data.get('email')}")
            print(f"First Name: {data.get('first_name')}")
            print(f"Last Name: {data.get('last_name')}")
            print(f"Gender: {data.get('gender')}")
            print(f"Birthday: {data.get('birthday')}")
            print(f"Location: {data.get('location')}")
            print(f"Work: {data.get('work')}")
            print(f"Education: {data.get('education')}")
            print(f"Relationship Status: {data.get('relationship_status')}")
            return data
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
    except facebook.GraphAPIError as e:
        print(f"Facebook API error: {e}")


def social_media_search(username, verbose=0):
    print(f"Searching for username: {username} on other social media platforms.")
    results = sherlock.sherlock(username, verbose=verbose)
    print(f"Found {len(results)} results:")
    for result in results:
        print(result)


def data_analysis(data, output=None):
    print("Analyzing and visualizing the data from Facebook account.")
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    # Transpose the DataFrame to have one row and multiple columns
    df = df.T
    # Drop the columns that are not relevant or have missing values
    df = df.drop(['id', 'email', 'name', 'first_name', 'last_name'], axis=1)
    # Convert the birthday column to datetime format
    df['birthday'] = pd.to_datetime(df['birthday'])
    # Extract the year from the birthday column and create a new column for age
    df['age'] = pd.datetime.now().year - df['birthday'].dt.year
    # Drop the birthday column
    df = df.drop('birthday', axis=1)
    # Convert the location column to a string and extract the city name
    df['location'] = df['location'].astype(str)
    df['location'] = df['location'].str.split("'name': ").str[1].str.split("'").str[0]
    # Convert the work column to a string and extract the employer name
    df['work'] = df['work'].astype(str)
    df['work'] = df['work'].str.split("'employer': ").str[1].str.split("'name': ").str[1].str.split("'").str[0]
    # Convert the education column to a string and extract the school name
    df['education'] = df['education'].astype(str)
    df['education'] = df['education'].str.split("'school': ").str[1].str.split("'name': ").str[1].str.split("'").str[0]
    # Print the DataFrame
    print(df)
    # Plot a pie chart of the gender distribution
    plt.figure()
    df['gender'].value_counts().plot(kind='pie', title='Gender Distribution', autopct='%1.1f%%')
    plt.show()
    # Plot a histogram of the age distribution
    plt.figure()
    df['age'].plot(kind='hist', title='Age Distribution', bins=10)
    plt.show()
    # Plot a bar chart of the location distribution
    plt.figure()
    df['location'].value_counts().plot(kind='bar', title='Location Distribution', rot=0)
    plt.show()
    # Plot a bar chart of the work distribution
    plt.figure()
    df['work'].value_counts().plot(kind='bar', title='Work Distribution', rot=0)
    plt.show()
    # Plot a bar chart of the education distribution
    plt.figure()
    df['education'].value_counts().plot(kind='bar', title='Education Distribution', rot=0)
    plt.show()
    # Optionally, save the DataFrame and the plots to a file in the specified format
    if output:
        # TODO: implement the output logic
        pass


def data_summary(data, output=None):
    print("Generating a summary of the data from Facebook account.")
    # Convert the data to a list of strings
    data_list = []
    for key, value in data.items():
        if isinstance(value, dict):
            value = value.get('name')
        data_list.append(str(value))
    # Create a gensim dictionary and corpus from the data list
    dictionary = gensim.corpora.Dictionary([data_list])
    corpus = [dictionary.doc2bow([text]) for text in data_list]
    # Create a gensim LDA model with 1 topic and 10 passes
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                        id2word=dictionary,
                                        num_topics=1,
                                        random_state=100,
                                        update_every=1,
                                        chunksize=100,
                                        passes=10,
                                        alpha='auto',
                                        per_word_topics=True)
    # Print the topic words and their probabilities
    print(lda_model.print_topics())
    # Optionally, save the summary to a file in the specified format
    if output:
        # TODO: implement the output logic
        pass


def main():
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description=f"{module_name} (Version {__version__})"
    )
    parser.add_argument(
        "account_id",
        help="Facebook account ID or URL to search for."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Display extra information."
    )
    parser.add_argument(
        "-o", "--output",
        choices=["csv", "json"],
        help="Specify output format."
    )
    parser.add_argument(
        "-s", "--search",
        action="store_true",
        help="Search for the given username on other social media platforms."
    )
    parser.add_argument(
        "-a", "--analyze",
        action="store_true",
        help="Analyze and visualize the data from Facebook account."
    )
    parser.add_argument(
        "-d", "--summary",
        action="store_true",
        help="Generate a summary of the data from Facebook account."
    )
    args = parser.parse_args()

    # Handle the receipt of a signal.
    signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))

    account_id = args.account_id
    verbose = args.verbose
    output = args.output
    search = args.search
    analyze = args.analyze
    summary = args.summary

    if account_id.isdigit():
        # Search for the Facebook account information
        data = facebook_advanced_search(account_id, verbose)
        if search:
            # Search for the given username on other social media platforms
            social_media_search(data.get('name'), verbose)
        if analyze:
            # Analyze and visualize the data from Facebook account
            data_analysis(data, output)
        if summary:
            # Generate a summary of the data from Facebook account
            data_summary(data, output)
    else:
        print("Please provide a valid Facebook account ID or URL.")


if __name__ == "__main__":
    main()
