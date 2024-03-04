import signal
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import sherlock
import gensim
import asyncio
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from colorama import init

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
    df['work'] = df['work'].str.split("'name': ").str[1].str.split("'").str[0]
    # Create a new column for education level based on the education column
    df['education_level'] = df['education'].apply(lambda x: gensim.summarization.keywords(x, words=1, lemmatize=True))

    # Plot a pie chart of the gender distribution
    gender_counts = df['gender'].value_counts()
    plt.figure()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
    plt.title('Gender Distribution')
    if output:
        plt.savefig('gender_distribution.png')  # Save the plot to a file
    else:
        plt.show()  # Show the plot on the screen

    # Plot a bar chart of the average age by relationship status
    age_mean = df.groupby('relationship_status')['age'].mean()
    plt.figure()
    plt.bar(age_mean.index, age_mean)
    plt.title('Average Age by Relationship Status')
    plt.xlabel('Relationship Status')
    plt.ylabel('Average Age')
    plt.xticks(rotation=45)
    if output:
        plt.savefig('age_by_relationship_status.png')  # Save the plot to a file
    else:
        plt.show()  # Show the plot on the screen

    # Create a summary of the data from the Facebook account
    summary = f"""
    Facebook Account Summary:
    Number of users: {len(df)}
    Average age: {df['age'].mean():.1f}
    Most common location: {df['location'].mode()[0]}
    Most common education level: {df['education_level'].mode()[0]}
    Most common relationship status: {df['relationship_status'].mode()[0]}
    Most common employer: {df['work'].mode()[0]}
    """
    print(summary)  # Print the summary to the console

    # Optionally, save the summary to a file in the specified format
    if output:
        with open('facebook_account_summary.txt', 'w') as f:
            f.write(summary)

async def main():
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

    if analyze or search:
        print("Analysis and search functionalities are disabled as the program does not use Facebook Graph API.")
        sys.exit(1)

    if summary:
        print("Summary functionality is disabled as the program does not use Facebook Graph API.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
