# app.py

from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from colorhash import ColorHash
import circlify
import json
from math import floor, ceil
import umap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from lime.lime_tabular import LimeTabularExplainer


def get_top_songs(df, month, top):
    df = df[df['month'] == month].copy()
    df = df[['master_metadata_track_name', 'master_metadata_album_artist_name', 'master_metadata_album_album_name', 'spotify_track_uri']].copy()
    counts = df['spotify_track_uri'].value_counts(sort=False).reset_index()
    df = df.drop_duplicates().reset_index().drop(columns=['index'])
    df['count'] = counts['spotify_track_uri']
    df = df.sort_values(by=['count'], ascending=False)
    top_tracks = [*df['master_metadata_track_name'][:top]]
    counts = [*df['count'][:top]]
    colors = []
    for i in top_tracks:
        colors.append(ColorHash(i).hex)
    
    data = {"tracks":top_tracks, "counts":counts, "colors":colors}
    
    return data


def get_top_artists(df, month, top):
    df = df[df['month'] == month].copy()
    df = df[['master_metadata_track_name', 'master_metadata_album_artist_name', 'master_metadata_album_album_name', 'spotify_track_uri']].copy()
    counts = df['master_metadata_album_artist_name'].value_counts(sort=False).reset_index()
    df = df.drop_duplicates(subset='master_metadata_album_artist_name').reset_index().drop(columns=['index'])
    df['count'] = counts['master_metadata_album_artist_name']
    df = df.sort_values(by=['count'], ascending=False)
    top_tracks = [*df['master_metadata_album_artist_name'][:top]]
    counts = [*df['count'][:top]]
    colors = []
    for i in top_tracks:
        colors.append(ColorHash(i).hex)
    
    data = {"tracks":top_tracks, "counts":counts, "colors":colors}
    
    return data

def do_random_forest(tracks, app):
    FEATURE_LIST = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo']
    df_features = pd.read_csv('data/data_elian_features.csv')
    df_features = df_features.filter(['master_metadata_track_name', 'master_metadata_album_artist_name', 'spotify_track_uri'] + FEATURE_LIST)

    # percentage of average/median to define if song = 'liked'
    percentage = 2
    median_freq = tracks['count'].median() # median freq
    threshold_freq = percentage * median_freq # threshold freq based on average/median
    tracks['liked'] = tracks['count'] > threshold_freq # liked=True when > threshold
    df_predict = pd.merge(df_features, tracks, on=['spotify_track_uri','master_metadata_track_name', 'master_metadata_album_artist_name'])
    df_predict = df_predict.drop(['count'], axis=1)

    # Define the target column of the model
    target = 'liked'                #could be genre?
    
    # Split the dataset into train and test set
    if len(df_predict) > 1:
        X_train, X_test, y_train, y_test = train_test_split(df_predict[FEATURE_LIST], df_predict[target], train_size=0.8, random_state=0)
    # Error catching
    else: 
        df = pd.DataFrame(0, index=np.arange(2), columns=['x', 'y'])
        fig = px.scatter(df,
            x='x', 
            y='y')
        fig.update_layout(title="There are too few samples for a random forest prediction")
        return fig, 0, 0

    # Instantiate a random forest classifier with 100 trees and a maximum depth of 5
    rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    # Train the random forest model on train set
    rfc.fit(X_train, y_train)

    #UMAP
    reducer = umap.UMAP(random_state=0)
    data = df_predict[FEATURE_LIST]
    data = data.drop_duplicates()
    scaled_data = StandardScaler().fit_transform(data)
    embedding = reducer.fit_transform(scaled_data)
    y_pred = rfc.predict_proba(data)
    result = pd.concat([df_features.drop_duplicates().reset_index(drop=True), pd.DataFrame(y_pred[:,1], columns=['like_prob'])], axis=1)
    result = pd.concat([result, pd.DataFrame(embedding, columns=['x', 'y'])], axis=1)
    result = result.round({'like_prob': 4})
    fig = px.scatter(result, 
            x='x', 
            y='y', 
            color='like_prob', 
            color_continuous_scale='speed',
            hover_data={'x':False, 
                        'y':False,
                        'like_prob':True,
                        'master_metadata_album_artist_name':True,
                        'master_metadata_track_name':True
                        })
    fig.for_each_trace(lambda t: t.update(hovertemplate = t.hovertemplate.replace('like_prob', 'Likeliness')))
    fig.for_each_trace(lambda t: t.update(hovertemplate = t.hovertemplate.replace('master_metadata_album_artist_name', 'Artist')))
    fig.for_each_trace(lambda t: t.update(hovertemplate = t.hovertemplate.replace('master_metadata_track_name', 'Track Name')))

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.layout.margin.b = 0
    fig.layout.margin.t = 40
    fig.layout.margin.l = 0
    fig.layout.margin.r = 0

    fig.layout.height = 350 # TODO: Tune height of the graph

    return fig, result, rfc

def lime_plot(x, y, result, rfc):
    # list with all song features
    FEATURE_LIST = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo']

    # define lime explainer
    lime_explainer = LimeTabularExplainer(
        result[FEATURE_LIST],
        mode="classification",
        feature_names=FEATURE_LIST,
        kernel_width=np.sqrt(len(result)),
        discretize_continuous=False,
        feature_selection="forward_selection",
    )

    # selected sample from random forest plot
    sample = result.loc[(result['x'] == x) & (result['y'] == y)]

    # Error catching
    if len(sample) == 0:
        return empty_lime()
    
    # explain the sample
    explanation = lime_explainer.explain_instance(
        sample[FEATURE_LIST].iloc[0],
        rfc.predict_proba,
        num_features=10,
        num_samples=5000,
        distance_metric="euclidean",
    )
    
    # create a barchart of the LIME features
    df = pd.DataFrame(np.array(explanation.as_list()), columns=['Features', 'Values'])
    df['Values'] = df['Values'].apply(lambda x: float(x))
    fig = px.bar(df, x='Values', y='Features', orientation='h', barmode='relative')

    fig.layout.margin.b = 0
    fig.layout.margin.t = 40
    fig.layout.margin.l = 0
    fig.layout.margin.r = 0

    fig.layout.height = 350 # TODO: Tune height of the graph

    return fig

# default empty lime plot
def empty_lime():
    # default axis values
    FEATURE_LIST = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo']
    values = [0 for i in FEATURE_LIST]

    # default plot
    df = pd.DataFrame(values, FEATURE_LIST).reset_index().rename(columns={0: "Values", "index": "Features"})
    fig = px.bar(df, x='Values', y='Features', orientation='h', barmode='relative')
    fig.update_layout(title="Activate LIME by selecting a scatter in the plot")

    fig.layout.margin.b = 0
    fig.layout.margin.t = 40
    fig.layout.margin.l = 0
    fig.layout.margin.r = 0

    fig.layout.height = 350 # TODO: Tune height of the graph

    return fig


# Authenticate using the Spotify server
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="fc126aaa02334aae871ae10bdba19854",
                                               client_secret="77feae55e0b949788ea1e3de052e4230",
                                               redirect_uri="http://localhost:8085/callback/",
                                               scope="user-library-read, user-top-read"))


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])


# Top 10 songs of user with graph for tempo and duration
top_tracks = sp.current_user_top_tracks(limit=10, time_range='short_term')
top_tracks = [{"title": track["name"], "artist": track["artists"][0]["name"], "id": track["id"]} for track in top_tracks["items"]]

top_tracks_features = sp.audio_features(tracks=[track["id"] for track in top_tracks])


# RadViz for two songs
index_first_song = 0
index_second_song = 1

categories = ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "valence"]

# Scale of the different categories, according to https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features
# acousticness: [0,1]
# danceability: [0,1]
# energy: [0,1]
# instrumentalness: [0,1]
# liveness: [0,1]
# loudness: typically [-60,0]
# speechiness: [0,1]
# valence: [0,1]
# --> Need to scale the loudness, using affine transformation

# new_loudness = (original_loudness + 60) * (1 / 60) + 0 = 1 + original_loudness / 60

### Visualisation of personal listening over time
# Getting the datasets
data_files = []
for data_folder in [folder for folder in os.listdir("data") if "." not in folder]:
    for file in os.listdir("data/{}".format(data_folder)):
        if file.endswith(".csv"):
            data_files.append({"value": "data/{}/{}".format(data_folder, file), "label": file.split("_")[1].capitalize()[:-4]})

# Setting the various options for the timespan
timespan_options = [{"value": "year", "label": "per year"}, {"value": "month", "label": "per month"}, {"value": "hour", "label": "per hour of the day"}]


app.layout = dbc.Container([

    dbc.Row([
        dbc.Col([
            # TODO: Add labels to the different options
            dcc.Dropdown(
                id="datasets-dropdown",
                options=data_files,
                value=data_files[0]["value"]
            ),

            dcc.Dropdown(
                id="timespan-dropdown",
                options=timespan_options,
                value=timespan_options[0]["value"]
            ),

            dbc.RadioItems(
                # TODO: Make this radio button nicer, for instance with a slider
                id="filter-column",
                options=[
                    {"label": "Artist", "value": "master_metadata_album_artist_name"},
                    {"label": "Song Title", "value": "master_metadata_track_name"}
                ],
                value="master_metadata_album_artist_name"
            ),

            dbc.Input(
                id="filter-value",
                type="text",
                placeholder="Filter"
            )
        ], width=2),

        dbc.Col([
            html.Div(id="top-tracks")
        ], width=3),

        dbc.Col([
            dcc.Graph(
                id="listening-duration-graph",
                config = {"modeBarButtonsToRemove": ["lasso2d"]}
            )
        ], width=7)

    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='random-forest',
            )
        ], width=8),

        dbc.Col([
            dcc.Graph(
                id='lime-graph',
                figure = empty_lime(),
            )
        ], width=4)
    ]),


    dcc.Store(id='dataset'),
    dcc.Store(id='slider-marks')
], style={"max-width": "100%", "paddingTop": "12px"})

@app.callback(
    Output('lime-graph', "figure"),
    Input("dataset", "data"),
    Input("random-forest", "clickData"),    
    Input("listening-duration-graph", "relayoutData"),
    Input("timespan-dropdown", "value"),    
    Input("filter-column", "value"),
    Input("filter-value", "value")
)
def click(data, clickData, graph_events, timespan, filter_column, filter):
    if not clickData:
        raise PreventUpdate

    df = pd.read_json(data)
    df.drop_duplicates(inplace=True)

    # TODO Change filter such that the name of the artist doesn't needs to be exactly correct
    if filter is not None:
        if filter != "":
            df = df[df[filter_column] == filter]

    # When the graph is first loaded or the scale is reset
    if "xaxis.autorange" in graph_events or "autosize" in graph_events:
        top_tracks = get_top_songs_range(df)
        _, result, rfc = do_random_forest(top_tracks, app)
        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']
        track_name = clickData['points'][0]['customdata'][2]
        artist = clickData['points'][0]['customdata'][1]
        fig = lime_plot(x, y, result, rfc)
        fig.update_layout(title="LIME plot for " + track_name + " from " + artist)

    # When the user has resized the graph
    if "xaxis.range[0]" in graph_events:
        app.logger.info(graph_events)
        top_tracks = get_top_songs_range(df, graph_events["xaxis.range[0]"], graph_events["xaxis.range[1]"], timespan)
        _, result, rfc = do_random_forest(top_tracks, app)
        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']
        track_name = clickData['points'][0]['customdata'][2]
        artist = clickData['points'][0]['customdata'][1]
        fig = lime_plot(x, y, result, rfc)
        fig.update_layout(title="LIME plot for " + track_name + " from " + artist)

    return fig

@app.callback(
    Output("dataset", "data"),
    Output("slider-marks", "data"),
    Input("datasets-dropdown", "value")
)
def load_dataset(path):

    # Load the data set
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)

    unique_months = df["month"].unique()

    # Compute the slider marks
    slider_marks = {}
    for index, month in enumerate(unique_months):
        slider_marks[index] = month

    return df.to_json(), json.dumps(slider_marks)


def get_top_songs_range(df, start_range=None, end_range=None, range_column=None):

    if start_range is not None:
        if range_column == "year" or range_column == "hour":
            start = ceil(start_range)
            end = floor(end_range)
            df = df[df[range_column].between(start, end)]
        if range_column == "month":
            months = df["month"].unique().tolist()

            start_index = months.index(start_range[:7])
            end_index = months.index(end_range[:7])

            months_to_filter = months[start_index : end_index+1]

            df = df[df[range_column].isin(months_to_filter)]

    tracks = df[['spotify_track_uri','master_metadata_track_name', 'master_metadata_album_artist_name', 'master_metadata_album_album_name']].copy()
    counts = df['spotify_track_uri'].value_counts(sort=False).reset_index()
    tracks = tracks.drop_duplicates().reset_index(drop=True)
    tracks['count'] = counts['spotify_track_uri']
    tracks.sort_values(by=['count'], ascending=False, inplace=True)
    tracks.reset_index(drop=True, inplace=True)
    return tracks


@app.callback(
    Output("top-tracks", "children"),
    Output("random-forest", "figure"),
    Input("dataset", "data"),
    Input("listening-duration-graph", "relayoutData"),
    Input("timespan-dropdown", "value"),
    Input("filter-column", "value"),
    Input("filter-value", "value")
)
def get_scale_graph(data, graph_events, timespan, filter_column, filter):

    if graph_events == {}:
        raise PreventUpdate

    df = pd.read_json(data)
    df.drop_duplicates(inplace=True)

    # TODO Change filter such that the name of the artist doesn't needs to be exactly correct
    if filter is not None:
        if filter != "":
            df = df[df[filter_column] == filter]

    # When the graph is first loaded or the scale is reset
    if "xaxis.autorange" in graph_events or "autosize" in graph_events:
        top_tracks = get_top_songs_range(df)

        top_songs_layout = convert_to_top_tracks_list(top_tracks.head(5))
        fig_rf, _, _ = do_random_forest(top_tracks, app)

        return top_songs_layout, fig_rf

    # When the user has resized the graph
    if "xaxis.range[0]" in graph_events:
        app.logger.info(graph_events)
        top_tracks = get_top_songs_range(df, graph_events["xaxis.range[0]"], graph_events["xaxis.range[1]"], timespan)

        top_songs_layout = convert_to_top_tracks_list(top_tracks.head(5))
        fig_rf, _, _ = do_random_forest(top_tracks, app)

        return top_songs_layout, fig_rf


def convert_to_top_tracks_list(data):
    layout = []
    for index, track in data.iterrows():
        album_cover = sp.track(track["spotify_track_uri"][14:])["album"]["images"][-1]["url"]
        song_tile = dbc.Row([
            dbc.Col([
                html.H3("#{}".format(index+1))
            ], width=1, class_name="p-0"),

            dbc.Col([
                html.Img(src=album_cover)
            ], width=2),

            dbc.Col([
                html.H5(track["master_metadata_track_name"]),
                html.Span(track["master_metadata_album_artist_name"])
            ], width=9, style={"paddingLeft": "24px"})
        ])

        layout.append(song_tile)
    return layout


@app.callback(
    Output("listening-duration-graph", "figure"),
    Input("dataset", "data"),
    Input("timespan-dropdown", "value"),
    Input("filter-column", "value"),
    Input("filter-value", "value")
)
def update_duration_listening_graph(data, timespan, filter_column, filter):

    df = pd.read_json(data)
    df.drop_duplicates(inplace=True)

    # TODO Change filter such that the name of the artist doesn't needs to be exactly correct
    if filter is not None:
        if filter != "":
            df = df[df[filter_column] == filter]

    duration = df.groupby(timespan)['ms_played'].sum().reset_index(name = 'Total duration')
    duration['hours'] = ((duration['Total duration'] / 1000) / 60) / 60
    fig = px.bar(duration, x=timespan, y='hours', title='Total duration per {}'.format(timespan)) \
                .update_xaxes(title = 'Date', visible = True, showticklabels = True) \
                .update_yaxes(title = 'Total hours', visible = True, showticklabels = True, fixedrange = True)

    fig.layout.margin.b = 0
    fig.layout.margin.t = 40
    fig.layout.margin.l = 0
    fig.layout.margin.r = 0
    fig.layout.height = 350 # TODO: Tune height of the graph

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True) # Running in debug mode makes the server reload automatically on changes

