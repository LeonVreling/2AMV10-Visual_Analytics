# app.py

from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import json
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import jsonpickle


def do_random_forest(tracks, features):
    #TODO - kijken naar threshold voor liked
    df_complete = pd.merge(tracks, features, on="spotify_track_uri")

    tracks['liked'] = tracks['count'] > 20 # liked=True when > threshold
    df_predict = pd.merge(df_complete, tracks[['spotify_track_uri', 'liked']], on=['spotify_track_uri'])

    # Define the target column of the model
    target = 'liked'                #could be genre?

    if len(df_predict) < 2: 
        fig = px.scatter()
        fig.update_layout(title="<b> There are too few samples for a random forest prediction <b>")
        return fig, pd.DataFrame(), 0

    # Instantiate a random forest classifier with 100 trees and a maximum depth of 5
    rfc = RandomForestClassifier(n_estimators=500, max_depth=50, random_state=0)
    # Train the random forest model on train set
    rfc.fit(df_predict[FEATURE_LIST].values, df_predict[target].values)

    data = df_predict[FEATURE_LIST]
    y_pred = rfc.predict_proba(data)
    # Error catching
    if y_pred.shape[1] < 2:
        fig = px.scatter()
        fig.update_layout(title="<b> There are too few samples for a random forest prediction <b>")
        return fig, pd.DataFrame(), 0
    
    reducer = umap.UMAP(random_state=0, init='random')
    scaled_data = StandardScaler().fit_transform(data)
    embedding = reducer.fit_transform(scaled_data)
    result = pd.concat([df_predict.reset_index(drop=True), 
                        pd.DataFrame(y_pred[:,1], columns=['like_prob'])], axis=1)
    result = pd.concat([result, pd.DataFrame(embedding, columns=['x', 'y'])], axis=1)
    result = result.round({'like_prob': 4})

    return result, rfc


def lime_plot(track_names, artists, result, rfc):
    # Define lime explainer
    lime_explainer = LimeTabularExplainer(
        result[FEATURE_LIST],
        mode="classification",
        feature_names=FEATURE_LIST,
        kernel_width=np.sqrt(len(result)),
        discretize_continuous=False,
        feature_selection="forward_selection",
    )

    # Selected sample from random forest plot
    sample = result.loc[(result['master_metadata_track_name'].isin(track_names)) & (result['master_metadata_album_artist_name'].isin(artists))]

    # Error catching
    if len(sample) == 0:
        return "Something went wrong, please try selecting a point again"
    
    explenations = []
    for _, element in sample.iterrows():
        # Explain the sample
        explanation = lime_explainer.explain_instance(
            element[FEATURE_LIST],
            rfc.predict_proba,
            num_features=len(FEATURE_LIST),
            num_samples=100,
            distance_metric="euclidean",
        )
        explenations.append(explanation.as_list())

    if len(explenations) > 1:
        explained_features = []
        reshaped_explenations = np.reshape(explenations, (-1, 2))
        keys = np.unique(reshaped_explenations[:, 0])

        # Calculate the average per key
        for key in keys:
            values = reshaped_explenations[reshaped_explenations[:, 0] == key, 1].astype(float)
            average = np.mean(values)
            explained_features.append([key, average])
    else:
        explained_features = explenations[0]
    
    # Create a barchart of the LIME features
    df = pd.DataFrame(sorted(explained_features,key=lambda x: x[0], reverse=True), columns=['Features', 'Values'])
    df['Values'] = df['Values'].apply(lambda x: float(x))
    fig = px.bar(df, x='Values', y='Features', color_discrete_sequence=px.colors.qualitative.Pastel1, orientation='h', barmode='relative')

    fig.layout.margin.b = 0
    fig.layout.margin.t = 40
    fig.layout.margin.l = 0
    fig.layout.margin.r = 0

    fig.layout.height = 350 # TODO: Tune height of the graph

    return fig


def pc_plot(result, rfc):
    y_pred = rfc.predict_proba(result[FEATURE_LIST])[:,1]
    result['pred'] = y_pred

    dimensions = sorted(FEATURE_LIST)
    
    fig = px.parallel_coordinates(result,
                                  color='pred',
                                  dimensions=dimensions + [("pred")],
                                  color_continuous_scale=px.colors.sequential.speed,
                                  color_continuous_midpoint=0.5)
    
    return fig


def new_top_songs(data, rfc, top_count):
    # Keep the songs that the user has not listened to
    X_test = pd.merge(all_features, data, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

    # Make predictions on the not yet heard songs
    y_pred = rfc.predict_proba(X_test[FEATURE_LIST])

    # Sort on highest like probability
    sort_pred = [x for _, x in sorted(zip(y_pred[:,0], list(X_test['spotify_track_uri'])))]

    # Create lists for top tracks and corresponding artists
    tracks = []
    artists = []

    i = 0
    j = len(tracks)
    # Continue until there are top_count unique songs recommended
    while j < top_count:
        # Retrieve track name and corresponding artist from Spotipy
        track_name = X_test[X_test['spotify_track_uri'] == sort_pred[i]]['master_metadata_track_name'].values[0]
        artist = X_test[X_test['spotify_track_uri'] == sort_pred[i]]['master_metadata_album_artist_name'].values[0]
        i = i + 1
        # Filter the songs that the user has not heard yet 
        # and make sure a song is not recommended twice (album vs single version)
        if len(data[data['master_metadata_track_name'] == track_name]) == 0 and track_name not in tracks:
            tracks.append(track_name)
            artists.append(artist)
            j = j + 1

    app.logger.info(tracks)
    app.logger.info(artists)

    return tracks, artists


# list with all song features
FEATURE_LIST = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo', 'time_signature', 'duration_ms']
all_features = pd.read_csv('data/data_features.csv').rename(columns={"uri": "spotify_track_uri"})

# Authenticate using the Spotify server
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="fc126aaa02334aae871ae10bdba19854",
                                               client_secret="77feae55e0b949788ea1e3de052e4230",
                                               redirect_uri="http://localhost:8085/callback/",
                                               scope="user-library-read, user-top-read"))


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
app.title = "SpotiFacts"


### Visualisation of personal listening over time
# Getting the datasets
data_files = []
for data_folder in [folder for folder in os.listdir("data") if "." not in folder]:
    for file in os.listdir("data/{}".format(data_folder)):
        if file.endswith(".csv") and "features" not in file:
            data_files.append({"value": "data/{}/{}".format(data_folder, file), "label": file.split("_")[1].capitalize()[:-4]})

# Setting the various options for the timespan
timespan_options = [{"value": "year", "label": "per year"}, {"value": "month", "label": "per month"}, {"value": "hour", "label": "per hour of the day"}]


app.layout = dbc.Container([

    dbc.Row([
        dbc.Col([
            html.Span("Dataset:"),

            dcc.Dropdown(
                id="datasets-dropdown",
                options=data_files,
                value=data_files[0]["value"],
                clearable=False
            ),

            dbc.RadioItems(
                id="filter-column",
                options=[
                    {"label": "Artist", "value": "master_metadata_album_artist_name"},
                    {"label": "Song Title", "value": "master_metadata_track_name"}
                ],
                value="master_metadata_album_artist_name",
                inline=True
            ),

            dbc.Input(
                id="filter-value",
                type="text",
                placeholder="Filter",
                debounce=True # Only execute callback on enter or losing focus
            ),

            html.Span("Group by:"),

            dcc.Dropdown(
                id="timespan-dropdown",
                options=timespan_options,
                value=timespan_options[0]["value"],
                clearable=False
            ),

            dcc.Graph(
                id="listening-duration-graph",
                config = {
                    "displayModeBar": False
                }
            ),

            html.Span("Amount of streams:"),

            # TODO: Change padding on the sides of the slider from 25px to 5px
            dcc.RangeSlider(
                id="streams-slider",
                min=0, 
                max=1000, # This values gets overwritten by a callback
                step=1,
                marks=None,
                value=[0, 1000], # The initial values of the handles
                tooltip={"placement": "bottom", "always_visible": True},
                allowCross=False # Make the handles not able to cross
            )      
        ], width=2),

        dbc.Col([
            html.H4("Your top songs"),
            html.Div(id="top-tracks")
        ], width=3),

        dbc.Col([
            dcc.Graph(
                id='PC-graph',
            )
        ], width=7)

    ]),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                    id="loading",
                    type="circle",
                    color="#DDDDDD",
                    children=dcc.Graph(id='random-forest-graph')
                ),
        ], width=6),

        dbc.Col([
            html.P("Select a point in the graph to see why the prediciton was made")
        ], width=3, id='lime-graph'),

        dbc.Col([
            html.H4("Predicted liked songs"),
            html.Div(id="predicted-tracks")
        ], width=3)
    ]),

    dcc.Store(id='full-dataset'),
    dcc.Store(id='filtered-dataset'),
    dcc.Store(id='filtered-dataset-by-streams'),
    dcc.Store(id='model'),
    dcc.Store(id='predictions'),
    dcc.Store(id='temp')

], style={"max-width": "100%", "paddingTop": "6px"})


def get_top_songs(df):

    tracks = df[['spotify_track_uri','master_metadata_track_name', 'master_metadata_album_artist_name', 'master_metadata_album_album_name']].copy()
    counts = df['spotify_track_uri'].value_counts(sort=False).reset_index()
    tracks = tracks.drop_duplicates().reset_index(drop=True)
    tracks['count'] = counts['spotify_track_uri']
    tracks.sort_values(by=['count'], ascending=False, inplace=True)
    tracks.reset_index(drop=True, inplace=True)
    return tracks


@app.callback(
    Output("full-dataset", "data"),
    Input("datasets-dropdown", "value")
)
def load_data(path):

    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)

    return df.to_json()


@app.callback(
    Output("filtered-dataset", "data"),
    Input("datasets-dropdown", "value"),
    Input("listening-duration-graph", "selectedData"),
    Input("timespan-dropdown", "value"),
    Input("filter-column", "value"),
    Input("filter-value", "value")
)
def load_and_filter_data(path, selection, timespan, filter_column, filter):

    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)

    if filter is not None:
        if filter != "":
            df = df[df[filter_column] == filter]

    if selection is not None and selection['points'] != []:
        selected_points = [point["x"] for point in selection['points']]
        df = df[df[timespan].isin(selected_points)]

    return df.to_json()


@app.callback(
    Output("filtered-dataset-by-streams", "data"),
    Input("filtered-dataset", "data"),
    Input("streams-slider", "value"),
    Input("streams-slider", "max")
)
def filter_data_by_streams(data, streams, max_streams):

    df = pd.read_json(data)

    if streams[0] != 0 or streams[1] != max_streams: # Prevent the filtering on initial loading
        tracks_with_count = get_top_songs(df)
        filtered_tracks_with_count = tracks_with_count.loc[tracks_with_count['count'].between(streams[0], streams[1]), 'spotify_track_uri'].tolist()
        df = df[df["spotify_track_uri"].isin(filtered_tracks_with_count)]
    
    return df.to_json()


@app.callback(
    Output("streams-slider", "max"),
    Output("streams-slider", "value"),
    Input("filtered-dataset", "data")
)
def get_highest_stream_count(data):
    df = pd.read_json(data)
    streams = get_top_songs(df)
    max_count = streams.head(1)["count"][0]
    return max_count, [0, max_count]


@app.callback(
    Output("model", "data"),
    Output("predictions", "data"),
    Input("full-dataset", "data"),
)
def train_random_forest(data):
    df = pd.read_json(data)

    # Get the features of the songs that are in the dataset
    features = pd.merge(all_features.drop(columns=['master_metadata_album_artist_name', 'master_metadata_track_name']), df, indicator=True, how='inner', on="spotify_track_uri")
    features = features[all_features.columns].drop(['Unnamed: 0', 'id', 'track_href', 'analysis_url', 'master_metadata_album_artist_name', 'master_metadata_track_name'], axis=1)
    features.drop_duplicates(inplace=True)

    top_tracks = get_top_songs(df)

    result, rfc = do_random_forest(top_tracks, features)

    return jsonpickle.encode(rfc), json.dumps(result.to_dict("index"))


@app.callback(
    Output("random-forest-graph", "figure"),
    Input("filtered-dataset-by-streams", "data"),
    Input("predictions", "data")
)
def show_random_forest(data, predictions):
    df = pd.read_json(data)
    result = pd.read_json(predictions, orient="index")

    # Get only the results from the selected points
    selected_points = result[result["spotify_track_uri"].isin(df["spotify_track_uri"])]

    #TODO ? af laten hangen van eventuele slider of helemaal weghalen
    result = result[result['count'] > 1]
    fig = px.scatter(selected_points, 
            x='x', 
            y='y', 
            color='like_prob', 
            color_continuous_scale='speed',
            color_continuous_midpoint=0.5,
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

    fig.update_coloraxes(colorbar_title_text="Likeliness")

    fig.layout.margin.b = 0
    fig.layout.margin.t = 0
    fig.layout.margin.l = 0
    fig.layout.margin.r = 0

    fig.layout.height = 325 # TODO: Tune height of the graph

    fig.update_layout(clickmode='event+select')

    return fig


@app.callback(
    Output("top-tracks", "children"),
    Input("filtered-dataset", "data")
)
def display_top_tracks(data):
    
    df = pd.read_json(data)

    top_tracks = get_top_songs(df).head(5)

    layout = []
    for index, track in top_tracks.iterrows():
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
    Output('lime-graph', "children"),
    Input("model", "data"),
    Input("predictions", "data"),
    Input("random-forest-graph", "selectedData")
)
def display_lime_plot(model, predictions, selection):
    if not selection:
        raise PreventUpdate

    rfc = jsonpickle.decode(model)
    result = pd.read_json(predictions, orient="index")

    selected_tracks = [point["customdata"][2] for point in selection["points"]]
    selected_artists = [point["customdata"][1] for point in selection["points"]]

    fig = lime_plot(selected_tracks, selected_artists, result, rfc)

    # Error handling
    if type(fig) == str:
        return html.P(fig)
    
    if len(selected_tracks) == 1:
        fig.update_layout(title="<b> LIME plot for <b>" + selected_tracks[0] + "<b> by <b>" + selected_artists[0])
    else:
        fig.update_layout(title="<b> LIME plot for selected cluster of <b>" + str(len(selected_tracks)) + "<b> songs <b>")

    return dcc.Graph(figure=fig)


@app.callback(
    Output("predicted-tracks", "children"),
    Input("full-dataset", "data"),
    Input("model", "data")
)
def predict_new_tracks(data, model):
    
    df = pd.read_json(data)
    rfc = jsonpickle.decode(model)

    AMOUNT_OF_PREDICTIONS = 10

    predicted_tracks, predicted_artists = new_top_songs(df, rfc, AMOUNT_OF_PREDICTIONS)

    predicted_songs_list = []

    # TODO: Make a nice layout to show the predicted top songs
    for i in range(AMOUNT_OF_PREDICTIONS):
        predicted_songs_list.append(
            html.Li(f"{predicted_tracks[i]} - {predicted_artists[i]}")
        )

    return html.Ul(children=predicted_songs_list)


@app.callback(
    Output('PC-graph', "figure"),
    Input("model", "data"),
    Input("predictions", "data"),
    Input("filtered-dataset", "data"),
    Input("random-forest-graph", "selectedData")
)
def display_pc_plot(model, predictions, data, selection):

    rfc = jsonpickle.decode(model)
    result = pd.read_json(predictions, orient="index")

    df = pd.read_json(data)
    filtered_tracks = result[result["spotify_track_uri"].isin(df["spotify_track_uri"])]

    if selection:
        selected_tracks = [point["customdata"][2] for point in selection["points"]]
        selected_artists = [point["customdata"][1] for point in selection["points"]]

        filtered_tracks = filtered_tracks.loc[(result['master_metadata_track_name'].isin(selected_tracks)) & (result['master_metadata_album_artist_name'].isin(selected_artists))]

    # Print the full PC plot if nothing is selected in the random forest plot
    if filtered_tracks.empty:
        return pc_plot(result, rfc)

    fig = pc_plot(filtered_tracks, rfc)

    fig.layout.margin.l = 30
    fig.layout.margin.b = 30

    fig.layout.height = 325 # TODO: Tune height of the graph

    # Error handling
    if type(fig) == str:
        return html.P(fig)

    return fig


@app.callback(
    Output("listening-duration-graph", "figure"),
    Input("datasets-dropdown", "value"),
    Input("timespan-dropdown", "value")
)
def update_duration_listening_graph(path, timespan):

    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)

    duration = df.groupby(timespan)['ms_played'].sum().reset_index(name = 'Total duration')
    duration['hours'] = ((duration['Total duration'] / 1000) / 60) / 60
    fig = px.bar(duration, x=timespan, y='hours', color_discrete_sequence=px.colors.qualitative.Pastel1) \
                .update_xaxes(title = 'Date', visible = True, showticklabels = True) \
                .update_yaxes(title = 'Total hours', visible = True, showticklabels = True, fixedrange = True)

    fig.layout.margin.b = 0
    fig.layout.margin.t = 0
    fig.layout.margin.l = 0
    fig.layout.margin.r = 0
    fig.layout.height = 125 # TODO: Tune height of the graph

    fig.update_layout(dragmode='select')

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True) # Running in debug mode makes the server reload automatically on changes

