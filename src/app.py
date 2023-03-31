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
from math import floor, ceil
import umap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import jsonpickle
from sklearn.preprocessing import LabelEncoder


def do_random_forest(tracks, features):
    #TODO ? slider voor minimaal aantal streams per liedje
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
    rfc.fit(df_predict[FEATURE_LIST], df_predict[target])

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
    #TODO ? af laten hangen van eventuele slider of helemaal weghalen
    result = result[result['count'] > 1]
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

    fig.update_coloraxes(colorbar_title_text="Likeliness")

    fig.layout.margin.b = 0
    fig.layout.margin.t = 40
    fig.layout.margin.l = 0
    fig.layout.margin.r = 0

    fig.layout.height = 350 # TODO: Tune height of the graph

    return fig, result, rfc


def lime_plot(track_name, artist, result, rfc):
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
    sample = result.loc[(result['master_metadata_track_name'] == track_name) & (result['master_metadata_album_artist_name'] == artist)]

    # Error catching
    if len(sample) == 0:
        return "Something went wrong, please try selecting a point again"
    
    # Explain the sample
    explanation = lime_explainer.explain_instance(
        sample[FEATURE_LIST].iloc[0],
        rfc.predict_proba,
        num_features=len(FEATURE_LIST),
        num_samples=100,
        distance_metric="euclidean",
    )
    
    # Create a barchart of the LIME features
    df = pd.DataFrame(sorted(np.array(explanation.as_list()),key=lambda x: x[0]), columns=['Features', 'Values'])
    df['Values'] = df['Values'].apply(lambda x: float(x))
    fig = px.bar(df, x='Values', y='Features', color_discrete_sequence=px.colors.qualitative.Pastel1, orientation='h', barmode='relative')

    fig.layout.margin.b = 0
    fig.layout.margin.t = 40
    fig.layout.margin.l = 0
    fig.layout.margin.r = 0

    fig.layout.height = 350 # TODO: Tune height of the graph

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
        track_name = sp.track(sort_pred[i])['name']
        artist = sp.track(sort_pred[i])['album']['artists'][0]['name']
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

            html.Span("Group by:"),

            dcc.Dropdown(
                id="timespan-dropdown",
                options=timespan_options,
                value=timespan_options[0]["value"],
                clearable=False
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
                placeholder="Filter",
                debounce=True # Only execute callback on enter or losing focus
            )
        ], width=2),

        dbc.Col([
            html.H4("Your top songs"),
            html.Div(id="top-tracks")
        ], width=3),

        dbc.Col([
            dcc.Graph(
                id="listening-duration-graph",
                config = {"modeBarButtonsToRemove": ["lasso2d","pan2d","select2d"]}
            )
        ], width=7)

    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='random-forest-graph',
            )
        ], width=6),

        dbc.Col([
            html.P("Select a point in the graph to see why the prediciton was made")
        ], width=3, id='lime-graph'),

        dbc.Col([
            html.H4("Predicted liked songs"),
            html.Div(id="predicted-tracks")
        ], width=3)
    ]),


    dcc.Store(id='dataset'),
    dcc.Store(id='model'),
    dcc.Store(id='predictions')

], style={"max-width": "100%", "paddingTop": "12px"})


@app.callback(
    Output('lime-graph', "children"),
    Input("model", "data"),
    Input("predictions", "data"),
    Input("random-forest-graph", "clickData"),
)
def click(model, predictions, clickEvent):
    if not clickEvent:
        raise PreventUpdate

    rfc = jsonpickle.decode(model)
    result = pd.read_json(predictions, orient="index")

    track_name = clickEvent['points'][0]['customdata'][2]
    artist = clickEvent['points'][0]['customdata'][1]
    fig = lime_plot(track_name, artist, result, rfc)

    # Error handling
    if type(fig) == str:
        return html.P(fig)

    fig.update_layout(title="<b> LIME plot for <b>" + track_name + "<b> by <b>" + artist)
    return dcc.Graph(figure=fig)


@app.callback(
    Output("dataset", "data"),
    Input("datasets-dropdown", "value")
)
def load_dataset(path):

    # Load the data set
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)

    return df.to_json()


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
    Output("random-forest-graph", "figure"),
    Output("model", "data"),
    Output("predictions", "data"),
    Output("predicted-tracks", "children"),
    Input("dataset", "data"),
    Input("listening-duration-graph", "relayoutData"),
    Input("timespan-dropdown", "value"),
    Input("filter-column", "value"),
    Input("filter-value", "value"),
    Input("datasets-dropdown", "value")
)
def get_scale_graph(data, graph_events, timespan, filter_column, filter, dataset_dropdown):

    if graph_events == {}:
        raise PreventUpdate

    df = pd.read_json(data)
    df.drop_duplicates(inplace=True)

    # Only get the features of songs that the selected person has listened to
    features = pd.merge(all_features, df, indicator=True, how='inner', on="spotify_track_uri")
    features = features[all_features.columns].drop(['Unnamed: 0', 'id', 'track_href', 'analysis_url'], axis=1)

    # TODO Change filter such that the name of the artist doesn't needs to be exactly correct
    if filter is not None:
        if filter != "":
            df = df[df[filter_column] == filter]

    # When the graph is first loaded or the scale is reset
    if "xaxis.autorange" in graph_events or "autosize" in graph_events:
        top_tracks = get_top_songs_range(df)

        top_songs_layout = convert_to_top_tracks_list(top_tracks.head(5))
        fig_rf, result, rfc = do_random_forest(top_tracks, features)

        AMOUNT_OF_PREDICTIONS = 10

        predicted_tracks, predicted_artists = new_top_songs(df, rfc, AMOUNT_OF_PREDICTIONS)

        predicted_songs_list = []

        # TODO: Make a nice layout to show the predicted top songs
        for i in range(AMOUNT_OF_PREDICTIONS):
            predicted_songs_list.append(
                html.Li(f"{predicted_tracks[i]} - {predicted_artists[i]}")
            )

        return top_songs_layout, fig_rf, jsonpickle.encode(rfc), json.dumps(result.to_dict("index")), html.Ul(children=predicted_songs_list)

    # When the user has resized the graph
    if "xaxis.range[0]" in graph_events:
        top_tracks = get_top_songs_range(df, graph_events["xaxis.range[0]"], graph_events["xaxis.range[1]"], timespan)

        top_songs_layout = convert_to_top_tracks_list(top_tracks.head(5))
        fig_rf, result, rfc = do_random_forest(top_tracks, features)

        return top_songs_layout, fig_rf, jsonpickle.encode(rfc), json.dumps(result.to_dict("index"))


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
    fig = px.bar(duration, x=timespan, y='hours', color_discrete_sequence=px.colors.qualitative.Pastel1, title='<b> Total duration per {} <b>'.format(timespan)) \
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

