## Generated all of the graphics in Graphs_3B

import pandas as pd
import matplotlib.pyplot as plt

def plot_by_genre(df):
    genre_counts = df["Genre"].value_counts()

    plt.bar(genre_counts.index, genre_counts.values)
    plt.xlabel("Genre")
    plt.ylabel("Number of Games")
    plt.title("Distribution of Genres")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_by_totalsales(df):
    plt.hist(df["Global_Sales"], bins=30, edgecolor='black', log=True)
    plt.xlabel("Global Sales (millions)")
    plt.ylabel("Number of Games")
    plt.title("Distribution of Global Sales")
    plt.show()

def plot_by_publisher(df, top_n = 20):
    publisher_counts = df["Publisher"].value_counts()
    top_publishers = publisher_counts.head(top_n)

    plt.figure(figsize=(8, 8))
    plt.barh(top_publishers.index[::-1], top_publishers.values[::-1])
    plt.xlabel("Number of Games")
    plt.ylabel("Publisher")
    plt.title(f"Top {top_n} Publishers by Number of Games")
    plt.tight_layout()
    plt.show()

def plot_bottom_publishers(df, bottom_n=20):
    publisher_counts = df["Publisher"].value_counts()

    bottom_publishers = publisher_counts.tail(bottom_n)

    plt.figure(figsize=(10, 12))
    plt.barh(bottom_publishers.index[::-1], bottom_publishers.values[::-1])
    plt.xlabel("Number of Games")
    plt.ylabel("Publisher")
    plt.title(f"Bottom {bottom_n} Publishers by Number of Games")
    plt.tight_layout()
    plt.show()

def plot_by_platform(df):
    platform_counts = df["Platform"].value_counts()
    plt.bar(platform_counts.index, platform_counts.values)
    plt.xlabel("Platform")
    plt.ylabel("Number of Games")
    plt.title("Distribution of Platforms")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_by_year(df):
    year_counts = df["Year"].value_counts()
    plt.bar(year_counts.index, year_counts.values)
    plt.xlabel("Year")
    plt.ylabel("Number of Games")
    plt.title("Distribution of Year")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def print_statistics(df):
    #general info
    total_games = len(df)
    total_genres = df["Genre"].nunique()
    total_publishers = df["Publisher"].nunique()
    min_year = df["Year"].min()
    max_year = df["Year"].max()
    print(f"Total games: {total_games}\nTotal genres: {total_genres}\nTotal publishers: {total_publishers}\nYear range: {int(min_year)}-{int(max_year)}")
    print()
    #mean, median, mode for sales
    regions = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
    for region in regions:
        mean_sales = df[region].mean()
        median_sales = df[region].median()
        mode_sales = df[region].mode()
        print(f"Sales statistics for {region} (in millions)")
        print(f"Mean global sales: {mean_sales:.3f}\nMedian global sales: {median_sales:.3f}\nMode global sales: {list(mode_sales)}\n")
    
    

if __name__ == "__main__":
    df = pd.read_csv("vgsales.csv")

    #plot_by_genre(df)
    #plot_by_totalsales(df)
    #plot_by_publisher(df, 75)
    #plot_bottom_publishers(df, 75)
    #plot_by_platform(df)
    #plot_by_year(df)
    print_statistics(df)