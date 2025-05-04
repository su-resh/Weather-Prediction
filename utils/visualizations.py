"""
This file deals with the visualization functions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import geoplot as gplt
import geoplot.crs as gcrs
import math

# Configure global plot style
plt.style.use('seaborn-v0_8')
sns.set_palette('viridis')

def plot_histogram(df, fig_size, subplot_rows, subplot_cols, cols_to_plot=None):
    ### Uni-variate visualization: Visualize histogram and boxplot of each variables
    ## visualize histogram

    if cols_to_plot is None:
        # select numeric columns
        # numeric_columns = df.select_dtypes(include=np.number).columns
        cols_to_plot = df.select_dtypes(include=np.number).columns
    
    ## check number of sub-plots
    if subplot_rows*subplot_cols < len(cols_to_plot):
        print('Number of subplots are insufficient.')
        print('Number of columns: ', len(cols_to_plot))
        print('Number of subplots: ', subplot_rows, ' x ', subplot_cols)
        return None
        
    fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=fig_size)
    
    ## first turn axis off all axs
    for ax in axs.flatten():
        ax.set_axis_off()
    # for ax in axs:
    #     if len(ax)>1:
    #         for axis in ax:
    #             axis.set_axis_off()
    #     else:
    #         ax.set_axis_off()
        
    row_index = 0
    col_index = 0
    for col in cols_to_plot:              
        # make required axis visible
        axs[row_index, col_index].set_axis_on()
        sns.histplot(df[col], ax=axs[row_index, col_index], kde=True)
#         ax=axs[row_index, col_index].title.set_text(col)
        col_index += 1
        if col_index == subplot_cols:
            col_index = 0
            row_index += 1
            
    fig.suptitle("Histogram and KDE plot.")
    plt.tight_layout()
    return fig

def plot_pairplot(df):
    ## Pair plot
    fig = sns.pairplot(df)
    return fig
    
def plot_correlation_heatmap(df, figsize=(6,6), annot=True):
    # select numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns
    corr_matrix = df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, fmt=".2f", cmap='coolwarm', square=True, ax=ax, cbar_kws={'shrink': 0.75})
    ax.set_title("Correlation Coefficient Heatmap", fontsize=10)

    return fig

def plot_district_map (gdf, title="Nepal District Map"):
    """
    Plot map of Nepal districts using gievn GeoDataFrame.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    gdf.plot(ax=ax, color='none', edgecolor='black', legend=True)
    plt.title("Nepal's Administrative Boundaries")
    ax.axis('off')
    plt.tight_layout()
    return plt

def choropleth_map (gdf_districtwise, column, title):
    """
    Choropleth in map of Nepal for given environment variable
    Parameters:
        gdf (_gdf_districtwise): A GeoDataFrame containing district geometries and a value column.
            By prefixing the parameter with an underscore, Streamlit knows not to hash or track it for cache invalidation.
        column (str): The name of the column to visualize.
        title (str): Title of the plot.
    """
    # fig, ax = plt.subplots(figsize=(10, 6))
    gplt.choropleth(
        gdf_districtwise, #GeoDataframe
        hue = column,  # column to visualize
        projection= gcrs.AlbersEqualArea(),
        legend=True,
        cmap='inferno_r',
        linewidth=0.5,    
        figsize=(12, 6),
        edgecolor='black',
        # ax=ax,
    )
    plt.title(title)
    plt.tight_layout()
    return plt

def plot_time_series(df, column, title):
    """
    plots time series data
    """
    fig, ax = plt.subplots(figsize = (12, 4))
    ax.plot(df['Date'], df[column])
    ax.set_xlabel("Date")
    ax.set_ylabel(column)
    ax.set_title(title)
    ax.grid(True)
    return fig

def plot_boxplot_monthly(df, column, title):
    """
    Plot monthwise boxplot to show seasonal trend
    """
    fig, ax = plt.subplots(figsize = (12, 4))
    sns.boxplot(x='Month', y=column, data=df, ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel(column)
    ax.set_title(title)
    ax.grid(True)
    # plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    #                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    return fig

def plot_event_type_distribution(df):
    """
    Plot barchart of climate EventType
    """
    event_counts = df['EventType'].value_counts()
    fig = px.pie(names=event_counts.index,
                 values=event_counts.values,
                 title="EventType Distribution",
                 hole=0.4  # for donut style (optional)
                 )
    return fig

def plot_event_type_districtwise(df):
    # Group by District and EventType
    district_event_counts = df.groupby(['District', 'EventType']).size().reset_index(name='count')
    # Sort by District alphabetically
    district_event_counts = district_event_counts.sort_values(by='District', ascending=False)

    # Group by EventType and sum 'count'
    event_type_counts = district_event_counts.groupby('EventType')['count'].sum().sort_values(ascending=False)
    # Get event types sorted by total count
    event_types = [event for event in event_type_counts.index.tolist() if event != 'Normal']

    # Create a 1-row, N-column subplot
    fig = make_subplots(
        rows=1, cols=len(event_types),
        subplot_titles=event_types,
        shared_yaxes=True,
        horizontal_spacing=0.02
    )

    # Add bar charts to each subplot
    for idx, event in enumerate(event_types):
        event_df = district_event_counts[district_event_counts['EventType'] == event]
        fig.add_trace(
            go.Bar(
                y=event_df['District'],
                x=event_df['count'],
                orientation='h',
                name=event,
                # text=event_df['count'],
                # textposition='outside'
            ),
            row=1,
            col=idx+1
        )

    # Update layout
    fig.update_layout(
        height=900,
        width=350 * len(event_types),  # Dynamic width based on number of event types
        showlegend=False,
        title_text="District-wise Extreme Event Counts",
    )
    # fig.update_xaxes(tickangle=-45)

    return fig

def plot_regression_evaluation(y_true, y_pred):
    """
    Plot Actual vs Predicted scatter plots for each target variable separately.
    Handles multi-output regression as well.
    """
    n_targets = y_true.shape[1]  # how many target columns
    subplot_cols = 3
    subplot_rows = math.ceil(n_targets/subplot_cols)
    figsize=(5 * subplot_cols, 4 * subplot_rows)
    fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=figsize)

    # ## first turn axis off all axs
    # for ax in axs.flatten():
    #     ax.set_axis_off()

    for i, ax in enumerate(axs.flatten()):
        if i<n_targets:
            # ax.set_axis_on()
            actual = y_true.iloc[:, i]
            predicted = y_pred.iloc[:, i]

            ax.scatter(actual, predicted, alpha=0.6, s=5)
            ax.plot(
                [actual.min(), actual.max()],  # x-coordinates
                [actual.min(), actual.max()],  # y-coordinates
                'k--', lw=2
            )  # reference line (45-degree diagonal) to compare actual vs predicted.
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Actual vs Predicted: {y_true.columns[i]}')
            ax.grid(True)
        else:
            ax.set_axis_off()

    plt.tight_layout()
    return plt

def plot_confusion_matrix(conf_matrix, fig_size = (4,3), class_names=None):
    plt.figure(figsize=fig_size)
    if class_names is not None:
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
    else:
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    return plt

def plot_regression_predictions(historical_df, predicted_df, district_encoded, district_name=None):
    """
    Plots historical and predicted climate variables in separate subplots.
    """
    # Filter for specific district
    historical = historical_df[
        (historical_df["district_encoded"] == district_encoded) &
        (historical_df["Date"] >= pd.to_datetime("2010-01-01"))
    ].copy()
    # historical = historical_df[historical_df["district_encoded"] == district_encoded].copy()
    predicted = predicted_df.copy()

    # Combine the data
    historical["source"] = "Historical"
    predicted["source"] = "Predicted"
    combined = pd.concat([historical, predicted], ignore_index=True)
    
    # Plot climate variables
    plt.figure(figsize=(12, 6))
    # variables_to_plot = ['Precip', 'Humidity_2m', 'Temp_2m', 'MinTemp_2m', 'MaxTemp_2m', ]
    variables_to_plot = {
        'Precip': 'mm',
        'Humidity_2m': '%',
        'Temp_2m': '°C',
        'MinTemp_2m': '°C',
        'MaxTemp_2m': '°C'
    }

    # Create figure with subplots
    fig, axes = plt.subplots(len(variables_to_plot), 1, figsize=(10, 4*len(variables_to_plot)))
    fig.suptitle(f"Climate Historical and Forecast Data for District {district_name or ''} ({district_encoded})", y=1.02)

    # Plot each variable in its own subplot
    for i, var in enumerate(variables_to_plot.items()):
        ax = axes[i]
        
        # Plot historical and predicted data
        ax.plot(historical['Date'], historical[var[0]], label="Historical", color='blue')
        ax.plot(predicted['Date'], predicted[var[0]], label="Predicted", color='orange')
        
        # Highlight prediction start
        ax.axvline(predicted["Date"].min(), color='red', linestyle='--', alpha=0.7)
        
        # Customize subplot
        ax.set_title(var[0])
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{var[0]} ({var[1]})")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    return fig