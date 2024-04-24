import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np

def univariate_eda_numerical(df, num_attribs):
    # Calculate the number of rows and columns for subplots
    num_plots = len(num_attribs)
    num_cols = math.ceil(math.sqrt(num_plots))
    num_rows = math.ceil(num_plots / num_cols)

    # Create subplots
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=num_attribs)

    # Iterate over numerical attributes and plot histograms
    for i, column in enumerate(num_attribs):
        row = i // num_cols + 1
        col = i % num_cols + 1
        trace = go.Histogram(x=df[column], name=column)
        fig.add_trace(trace, row=row, col=col)

    # Update layout
    fig.update_layout(height=600, width=800, title_text="Histograms of Numerical Attributes")
    return fig

def univariate_eda_categorical(df, cat_attribs):
    # Calculate the number of rows and columns for subplots
    num_plots = len(cat_attribs)
    num_cols = math.ceil(math.sqrt(num_plots))
    num_rows = math.ceil(num_plots / num_cols)

    # Create subplots
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=cat_attribs)

    # Iterate over categorical attributes and plot bar graphs
    for i, column in enumerate(cat_attribs):
        row = i // num_cols + 1
        col = i % num_cols + 1
        counts = df[column].value_counts()
        trace = go.Bar(x=counts.index, y=counts.values, name=column)
        fig.add_trace(trace, row=row, col=col)
        fig.update_xaxes(showticklabels=False, row=row, col=col)
        fig.update_yaxes(showgrid=False, showline=True, zeroline=False, row=row, col=col)
        fig.update_traces(showlegend=False)

    # Update layout
    fig.update_layout(height=900, width=800, title_text="Bar Graphs of Categorical Attributes",
                      legend=dict(x=0, y=-0.3))  # Adjust legend position
    return fig

def bivariate_eda_distribution(df, columns, hue_columns):
    """
    Plot distribution curves for each column in the DataFrame
    with respect to the specified hue columns using different colors.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the data.
        columns (list): List of columns to plot.
        hue_columns (list): List of columns to use for coloring the distribution curves.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    num_plots = len(columns) * len(hue_columns)
    fig, axes = plt.subplots(nrows=round(len(hue_columns)/2), ncols=2, figsize=(30, 20))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

    for i, hue_col in enumerate(hue_columns):
        unique_values = df[hue_col].unique()
        colors = sns.color_palette("husl", len(unique_values))

        for j, col in enumerate(columns):
            for k, value in enumerate(unique_values):
                sns.kdeplot(data=df[df[hue_col] == value], x=col, color=colors[k], ax=axes[i*len(columns) + j])
            axes[i*len(columns) + j].set_title(f"Distribution of {col} by {hue_col}")
            axes[i*len(columns) + j].set_xlabel(col)
            axes[i*len(columns) + j].set_ylabel('Density')
            
            # Get legend labels and sort them
            legend_labels = df[hue_col].value_counts().index.tolist()
            legend_labels.sort()
            
            axes[i*len(columns) + j].legend(legend_labels, title=hue_col)

    plt.tight_layout() 
    return fig

def multivariate_eda_heatmap(df, columns):
    """
    Create a heatmap of the correlation matrix for the specified columns in the DataFrame.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the data.
        columns (list): List of columns for which to create the heatmap.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Create a matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Plot the heatmap onto the figure, showing only the lower triangle
    corr_matrix = df[columns].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    fig = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=mask)
    
    plt.title('Correlation Heatmap')
    return fig.figure