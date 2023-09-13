import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go


def missing_control_plot(df, streamlit=False):
    """
    Generates a bar chart to visualize the number of missing values in each column of the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame to be analyzed for missing values.
        streamlit (bool, optional): If True, the chart will be displayed using Streamlit's st.plotly_chart().
                                    If False, the chart will be displayed using Plotly's fig.show().
                                    Default is False.

    Returns:
        missing_df (DataFrame): A DataFrame containing columns with missing values and their corresponding
                                counts of null values.
    """
    df_null = df.isna().sum()
    missing_df = pd.DataFrame(
        data=[df_null],
        columns=df.columns,
        index=["Null Size"]).T.sort_values("Null Size", ascending=False)
    missing_df = missing_df.loc[(missing_df["Null Size"] > 0)]
    fig = px.bar(missing_df, x=missing_df.index, y="Null Size", hover_name='Null Size',
                 color='Null Size', labels={
            "index": "Columns",
        },
                 color_discrete_sequence=['#D81F26'], template='plotly_dark',
                 title="Dataset Null Graph", width=1400, height=700)
    fig.update_layout(barmode='group')
    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)
    return missing_df


def missing_count_plot(df, missing_df, variable_type, streamlit=False):
    """
    Generates bar charts to visualize the count of missing values in categorical or numerical columns of the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame to be analyzed for missing values.
        missing_df (DataFrame): A DataFrame containing columns with missing values and their corresponding counts of null values.
        variable_type (str): The type of variables to visualize.
                             'num' for numerical columns and 'cat' for categorical columns.
        streamlit (bool, optional): If True, the charts will be displayed using Streamlit's st.plotly_chart().
                                    If False, the charts will be displayed using Plotly's bar.show().
                                    Default is False.
    """
    df_missing = df[missing_df.index]
    if variable_type == 'num':
        liste = df_missing.loc[:, df_missing.dtypes != "object"].columns
        template = 'plotly_dark'
    if variable_type == 'cat':
        liste = df_missing.loc[:, df_missing.dtypes == "object"].columns
        template = 'ggplot2'
    for i in liste:
        x = df[i].unique()
        y = [df[i][df[i] == j].count() for j in x]
        bar = px.bar(df, x=x, y=y, text=y, labels={
            "x": i,
            "y": "count",
        },
                     color_discrete_sequence=['#D81F26'], color=y,
                     template=template, title="Null Variable Count",
                     width=1000, height=500)
        if not streamlit:
            bar.show()
        else:
            st.plotly_chart(bar, use_container_width=True)


def corr_plot(df, target, streamlit=False):
    """
    This function generates a bar chart to visualize the correlation between a target column and other numerical columns in the DataFrame.

    Parameters:
        df (pandas DataFrame): The DataFrame from which the correlation will be calculated and visualized.
        target (str): The name of the target column for which the correlation will be measured against other numerical columns.
        streamlit (bool, optional): If True, the bar chart will be displayed using Streamlit's st.plotly_chart().
                                    If False, the bar chart will be displayed using Plotly's fig.show(). Default is False.
    Returns:
        This function does not return any value. It is used to generate and display the bar chart to visualize the correlation.

    """
    corr = df.select_dtypes(exclude=['object']).iloc[:, 1:].corr()
    corr_list = corr[target].sort_values(axis=0, ascending=False).iloc[1:]
    corr_df = pd.DataFrame(corr_list)
    fig = px.bar(corr_df, x=corr_df.index, y=target, hover_name=target,
                 color=target, labels={
            "index": "Columns",
            target: target,
        },
                 color_discrete_sequence=['#D81F26'], template='plotly_dark',
                 title="Dataset Correlation Graph", width=1400, height=700)
    fig.update_layout(barmode='group')
    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)


def line_chart(df, streamlit=False):
    """
    This function generates a line chart to visualize two series ('real' and 'pred') from the provided DataFrame.

    Parameters:
        df (pandas DataFrame): The DataFrame containing the data to be visualized.
        streamlit (bool, optional): If True, the line chart will be displayed using Streamlit's st.plotly_chart().
                                    If False, the line chart will be displayed using Plotly's fig.show(). Default is False.
    Returns:
        This function does not return any value. It is used to generate and display the line chart.

    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['real'], mode='lines', name='real', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['pred'], mode='lines', name='pred', line=dict(color='red')))
    fig.update_layout(title='Validation Predict', xaxis_title='Row', yaxis_title='Value')
    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)
