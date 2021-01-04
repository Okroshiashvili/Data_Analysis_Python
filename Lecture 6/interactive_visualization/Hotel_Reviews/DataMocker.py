


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.tools import mpl_to_plotly

import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# Read data
df = pd.read_json('hotel_reviews.json', orient='records', lines=True)

df_pos = df[df['sentiment'] == 'positive']
df_neg = df[df['sentiment'] == 'negative']
df_neu = df[df['sentiment'] == 'neutral']

df.loc[:, 'month_year'] = df['date'].dt.strftime('%B-%Y')
pos_count = df[df['sentiment'] == 'positive']['date'].value_counts().reset_index()
neg_count = df[df['sentiment'] == 'negative']['date'].value_counts().reset_index()
neu_count = df[df['sentiment'] == 'neutral']['date'].value_counts().reset_index()
date=df['date'].sort_values().dt.strftime('%B-%Y').unique().tolist()

cleanliness_count = df[df['aspect'] == 'Cleanliness']['date'].value_counts().reset_index()
service_count = df[df['aspect'] == 'Staff/service']['date'].value_counts().reset_index()
location_count = df[df['aspect'] == 'Location']['date'].value_counts().reset_index()
facilities_count = df[df['aspect'] == 'Facilities']['date'].value_counts().reset_index()





### Social Network Distribution ###

fig_1 = go.Figure(data=[go.Pie(labels=sorted(df['social-network'].unique()),
                             values=df['social-network'].value_counts().sort_index().tolist(),
                             text=sorted(df['social-network'].unique()),
                             textposition='auto',
                             hoverinfo='label+percent')])

fig_1.update_layout(title={'text':'Social Network Distribution',
                           'y':0.9,
                           'x':0.5,
                           'xanchor':'center',
                           'yanchor':'top'},
                    width=900,
                    height=700,
                    font=dict(family="Courier New, monospace",
                            size=14,
                            color="#111211")
                )




### Sentiment Distribution by Social Network ###

fig_2 = go.Figure(data=[

    go.Bar(name='Booking',
           x=sorted(df['sentiment'].unique().tolist()),
           y=df[df['social-network'] == 'booking']['sentiment'].value_counts().sort_index(),
           text=df[df['social-network'] == 'booking']['sentiment'].value_counts().sort_index(),
           textposition='auto'),
    
    go.Bar(name='Facebook',
           x=sorted(df['sentiment'].unique().tolist()),
           y=df[df['social-network'] == 'facebook']['sentiment'].value_counts().sort_index(),
           text=df[df['social-network'] == 'facebook']['sentiment'].value_counts().sort_index(),
           textposition='auto'),
    
    go.Bar(name='Instagram',
           x=sorted(df['sentiment'].unique().tolist()),
           y=df[df['social-network'] == 'instagram']['sentiment'].value_counts().sort_index(),
           text=df[df['social-network'] == 'instagram']['sentiment'].value_counts().sort_index(),
           textposition='auto'),

    go.Bar(name='TripAdvisor',
           x=sorted(df['sentiment'].unique().tolist()),
           y=df[df['social-network'] == 'tripadvisor']['sentiment'].value_counts().sort_index(),
           text=df[df['social-network'] == 'tripadvisor']['sentiment'].value_counts().sort_index(),
           textposition='auto'),

    go.Bar(name='Twitter',
           x=sorted(df['sentiment'].unique().tolist()),
           y=df[df['social-network'] == 'twitter']['sentiment'].value_counts().sort_index(),
           text=df[df['social-network'] == 'twitter']['sentiment'].value_counts().sort_index(),
           textposition='auto'),

    go.Bar(name='Yelp',
           x=sorted(df['sentiment'].unique().tolist()),
           y=df[df['social-network'] == 'yelp']['sentiment'].value_counts().sort_index(),
           text=df[df['social-network'] == 'yelp']['sentiment'].value_counts().sort_index(),
           textposition='auto'),
])

fig_2.update_layout(barmode='stack',
                  xaxis_title="Sentiments", 
                  yaxis_title="Counts",
                  title={'text':'Sentiment Distribution by Social Network',
                         'y':0.9,
                         'x':0.5,
                         'xanchor':'center',
                         'yanchor':'top'},
                  width=950,
                  height=700,
                  font=dict(family="Courier New, monospace",
                            size=14,
                            color="#111211")
                    )




### Emotion Distribution ###

fig_3 = make_subplots(rows=1,
                    cols=2,
                    specs=[[{'type':'domain'}, {'type':'domain'}]])

fig_3.add_trace(go.Pie(labels=sorted(df_pos['deep_sentiment'].unique().tolist()),
                     values=df_pos['deep_sentiment'].value_counts().sort_index(),
                     name='Positive',
                     text=sorted(df_pos['deep_sentiment'].unique().tolist()),
                     textposition='auto'), 1, 1)

fig_3.add_trace(go.Pie(labels=sorted(df_neg['deep_sentiment'].unique().tolist()),
                     values=df_neg['deep_sentiment'].value_counts().sort_index(),
                     name='Negative',
                     text=sorted(df_neg['deep_sentiment'].unique().tolist()),
                     textposition='auto'), 1, 2)

# Use `hole` to create a donut-like pie chart
fig_3.update_traces(hole=.5)

fig_3.update_layout(title={'text':'Emotion Distribution',
                         'y':0.9,
                         'x':0.5,
                         'xanchor':'center',
                         'yanchor':'top'},
                  width=900,
                  height=700,
                  font=dict(family="Courier New, monospace",
                            size=14,
                            color="#111211"),
                  annotations=[dict(text='Positive', x=0.15, y=0.5, font_size=20, showarrow=False),
                               dict(text='Negative', x=0.85, y=0.5, font_size=20, showarrow=False)]
                )




### Aspect Distribution ###

fig_4 = go.Figure(data=[

    go.Bar(name='Aspect',
           x=sorted(df['aspect'].unique()),
           y=df['aspect'].value_counts().sort_index(),
           text=df['aspect'].value_counts().sort_index(),
           textposition='auto')
])

fig_4.update_layout(title={'text':'Aspect Distribution',
                           'y':0.9,
                           'x':0.5,
                           'xanchor':'center',
                           'yanchor':'top'},
                    width=900,
                    height=700,
                    font=dict(family="Courier New, monospace",
                            size=14,
                            color="#111211")
                    )




### Sentiment Distribution by Aspect ###

fig_5 = go.Figure(data=[
    go.Bar(name='Positive',
           x=sorted(df['aspect'].unique().tolist()),
           y=pd.Series(df[df['sentiment'] == 'positive']['aspect'].value_counts()).sort_index().tolist(),
           text=pd.Series(df[df['sentiment'] == 'positive']['aspect'].value_counts()).sort_index().tolist(),
           textposition='auto'),
    
    go.Bar(name='Negative',
           x=sorted(df['aspect'].unique().tolist()),
           y=pd.Series(df[df['sentiment'] == 'negative']['aspect'].value_counts()).sort_index().tolist(),
           text=pd.Series(df[df['sentiment'] == 'negative']['aspect'].value_counts()).sort_index().tolist(),
           textposition='auto'),
    
    go.Bar(name='Neutral',
           x=sorted(df['aspect'].unique().tolist()),
           y=pd.Series(df[df['sentiment'] == 'neutral']['aspect'].value_counts()).sort_index().tolist(),
           text=pd.Series(df[df['sentiment'] == 'neutral']['aspect'].value_counts()).sort_index().tolist(),
           textposition='auto')
])

fig_5.update_layout(barmode='stack',
                  xaxis_title="Review Topics", 
                  yaxis_title="Counts",
                  title={'text':'Sentiment Distribution by Aspect',
                         'y':0.9,
                         'x':0.5,
                         'xanchor':'center',
                         'yanchor':'top'},
                  width=900,
                  height=700,
                  font=dict(family="Courier New, monospace",
                            size=14,
                            color="#111211")
                    )




### Adjective Distribution by Sentiment ###

fig_6 = make_subplots(rows=1, cols=2)

fig_6.add_trace(go.Bar(x=sorted(df_pos['adjective'].unique().tolist()),
                     y=df_pos['adjective'].value_counts().sort_index(),
                     name='Positive',
                     text=df_pos['adjective'].value_counts().sort_index(),
                     textposition='auto'), 1, 1)

fig_6.add_trace(go.Bar(x=sorted(df_neg['adjective'].unique().tolist()),
                     y=df_neg['adjective'].value_counts().sort_index(),
                     name='Negative',
                     text=df_neg['adjective'].value_counts().sort_index(),
                     textposition='auto'), 1, 2)


fig_6.update_layout(title={'text':'Adjective Distribution by Sentiment',
                           'y':0.9,
                           'x':0.5,
                           'xanchor':'center',
                           'yanchor':'top'},
                    width=950,
                    height=700,
                    font=dict(family="Courier New, monospace",
                            size=14,
                            color="#111211")
                    )




### Monthly Sentiment Distribution ###

fig_7 = go.Figure(data=[
    go.Bar(name='Positive',
           x=date,
           y=pos_count.groupby(pos_count['index'].dt.to_period('M')).sum()['date'].tolist(),
           text=pos_count.groupby(pos_count['index'].dt.to_period('M')).sum()['date'].tolist(),
           textposition='auto'),
    
    go.Bar(name='Negative',
           x=date,
           y=neg_count.groupby(neg_count['index'].dt.to_period('M')).sum()['date'].tolist(),
           text=neg_count.groupby(neg_count['index'].dt.to_period('M')).sum()['date'].tolist(),
           textposition='auto'),
    
    go.Bar(name='Neutral',
           x=date,
           y=neu_count.groupby(neu_count['index'].dt.to_period('M')).sum()['date'].tolist(),
           text=neu_count.groupby(neu_count['index'].dt.to_period('M')).sum()['date'].tolist(),
           textposition='auto'),
    ])


fig_7.update_layout(barmode='stack', 
                  yaxis_title="Counts",
                  title={'text':'Monthly Sentiment Distribution',
                         'y':0.9,
                         'x':0.5,
                         'xanchor':'center',
                         'yanchor':'top'},
                  width=950,
                  height=700,
                  font=dict(family="Courier New, monospace",
                            size=14,
                            color="#111211")
                    )


# Add slider
fig_7.update_xaxes(rangeslider_visible=True)



### Percentage Change of Sentiments ###

fig_8 = go.Figure()

fig_8.add_trace(go.Scatter(x=df['date'].sort_values().dt.strftime('%B-%Y').unique().tolist(),
                         y=(pos_count.groupby(pos_count['index'].dt.to_period('M')).sum()['date'].pct_change()*100).tolist(),
                    mode='lines+markers',
                    name='Positive'))

fig_8.add_trace(go.Scatter(x=df['date'].sort_values().dt.strftime('%B-%Y').unique().tolist(),
                         y=(neg_count.groupby(neg_count['index'].dt.to_period('M')).sum()['date'].pct_change()*100).tolist(),
                    mode='lines+markers',
                    name='Negative'))

fig_8.add_trace(go.Scatter(x=df['date'].sort_values().dt.strftime('%B-%Y').unique().tolist(),
                         y=(neu_count.groupby(neg_count['index'].dt.to_period('M')).sum()['date'].pct_change()*100).tolist(),
                    mode='lines+markers',
                    name='Neutral'))


fig_8.update_layout(yaxis_title="%",
                  title={'text':'Percentage Change of Sentiments',
                         'y':0.9,
                         'x':0.45,
                         'xanchor':'center',
                         'yanchor':'top'},
                  width=900,
                  height=600,
                  font=dict(family="Courier New, monospace",
                            size=14,
                            color="#111211")
                    )



### Monthly Aspect Distribution ###

fig_9 = go.Figure(data=[
    go.Bar(name='Cleanliness',
           x=date,
           y=cleanliness_count.groupby(cleanliness_count['index'].dt.to_period('M')).sum()['date'].tolist(),
           text=cleanliness_count.groupby(cleanliness_count['index'].dt.to_period('M')).sum()['date'].tolist(),
           textposition='auto'),
    
    go.Bar(name='Staff/service',
           x=date,
           y=service_count.groupby(service_count['index'].dt.to_period('M')).sum()['date'].tolist(),
           text=service_count.groupby(service_count['index'].dt.to_period('M')).sum()['date'].tolist(),
           textposition='auto'),
    
    go.Bar(name='Location',
           x=date,
           y=location_count.groupby(location_count['index'].dt.to_period('M')).sum()['date'].tolist(),
           text=location_count.groupby(location_count['index'].dt.to_period('M')).sum()['date'].tolist(),
           textposition='auto'),

    go.Bar(name='Facilities',
           x=date,
           y=facilities_count.groupby(facilities_count['index'].dt.to_period('M')).sum()['date'].tolist(),
           text=facilities_count.groupby(facilities_count['index'].dt.to_period('M')).sum()['date'].tolist(),
           textposition='auto')
])

fig_9.update_layout(barmode='stack', 
                  yaxis_title="Counts",
                  title={'text':'Monthly Aspect Distribution',
                         'y':0.9,
                         'x':0.5,
                         'xanchor':'center',
                         'yanchor':'top'},
                  width=950,
                  height=700,
                  font=dict(family="Courier New, monospace",
                            size=14,
                            color="#111211")
                    )


# Add Slider
fig_9.update_xaxes(rangeslider_visible=True)



### Percentage Change of Aspects ###

fig_10 = go.Figure()

fig_10.add_trace(go.Scatter(x=df['date'].sort_values().dt.strftime('%B-%Y').unique().tolist(),
                         y=(cleanliness_count.groupby(cleanliness_count['index'].dt.to_period('M')).sum()['date'].pct_change()*100).tolist(),
                    mode='lines+markers',
                    name='Cleanliness'))

fig_10.add_trace(go.Scatter(x=df['date'].sort_values().dt.strftime('%B-%Y').unique().tolist(),
                         y=(service_count.groupby(service_count['index'].dt.to_period('M')).sum()['date'].pct_change()*100).tolist(),
                    mode='lines+markers',
                    name='Staff/service'))

fig_10.add_trace(go.Scatter(x=df['date'].sort_values().dt.strftime('%B-%Y').unique().tolist(),
                         y=(location_count.groupby(location_count['index'].dt.to_period('M')).sum()['date'].pct_change()*100).tolist(),
                    mode='lines+markers',
                    name='Location'))

fig_10.add_trace(go.Scatter(x=df['date'].sort_values().dt.strftime('%B-%Y').unique().tolist(),
                         y=(facilities_count.groupby(facilities_count['index'].dt.to_period('M')).sum()['date'].pct_change()*100).tolist(),
                    mode='lines+markers',
                    name='Facilities'))


fig_10.update_layout(yaxis_title="%",
                  title={'text':'Percentage Change of Aspects',
                         'y':0.9,
                         'x':0.45,
                         'xanchor':'center',
                         'yanchor':'top'},
                  width=900,
                  height=600,
                  font=dict(family="Courier New, monospace",
                            size=14,
                            color="#111211")
                    )



### Review Count Change ###

fig_11 = go.Figure(go.Waterfall(
    name = "20",
    orientation = "v",
    x = df['date'].sort_values().dt.strftime('%B-%Y').unique().tolist()[1:],
    y = df.groupby(df['date'].dt.to_period('M')).count()['sentiment'].diff().sort_index().tolist()[1:],
    text = df.groupby(df['date'].dt.to_period('M')).count()['sentiment'].diff().sort_index().tolist()[1:],
    base = 2691,
    textposition = "outside",
))


fig_11.update_layout(title={'text':'Review Count Change',
                           'y':0.9,
                           'x':0.5,
                           'xanchor':'center',
                           'yanchor':'top'},
                    width=950,
                    height=700,
                    font=dict(family="Courier New, monospace",
                            size=14,
                            color="#111211"),
                    waterfallgroupgap = 0.1,
                    yaxis_title="Review Count"
                )




#####    Plot Side by Side     ######

app.layout = html.Div(children=[
    html.Div(
        dcc.Graph(
            figure= fig_1,
            style={'width': '800'}
        ), style={'display': 'inline-block'}),
    html.Div(
        dcc.Graph(
            figure=fig_2,
            style={'width': '800'}
        ), style={'display': 'inline-block'}),
    
    html.Div(
        dcc.Graph(
            figure=fig_3)
        , style={'display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(
            figure=fig_4),
        style={'display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(
            figure=fig_5),
        style={'display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(
            figure=fig_6),
        style={'display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(
            figure=fig_7),
        style={'display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(
            figure=fig_8),
        style={'display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(
            figure=fig_9),
        style={'display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(
            figure=fig_10),
        style={'display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(
            figure=fig_11),
        style={'display': 'inline-block'}
    )

], style={'width': '100%', 'display': 'inline-block'})



if __name__ == '__main__':
    app.run_server(debug=True)
