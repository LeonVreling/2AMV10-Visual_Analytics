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
# import operator


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

fig_radviz = go.Figure()

fig_radviz.add_trace(go.Scatterpolar(
      r=[top_tracks_features[index_first_song][category] if category != "loudness" else 1+top_tracks_features[index_first_song]["loudness"]/60 for category in categories],
      theta=categories,
      fill='toself',
      name= top_tracks[index_first_song]["title"] + " - " + top_tracks[index_first_song]["artist"]
))
fig_radviz.add_trace(go.Scatterpolar(
      r=[top_tracks_features[index_second_song][category] if category != "loudness" else 1+top_tracks_features[index_second_song]["loudness"]/60 for category in categories],
      theta=categories,
      fill='toself',
      name= top_tracks[index_second_song]["title"] + " - " + top_tracks[index_second_song]["artist"]
))

fig_radviz.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
  showlegend=True,
  title="Similarness of two songs"
)


### Visualisation of personal listening over time
# Getting the datasets
data_files = []
for data_folder in os.listdir("data"):
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
                id='radviz-example-graph',
                figure=fig_radviz
            )
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Slider(
                step=1,
                updatemode='drag',
                id='slider-bubble'
            ),

            dcc.Graph(id="bubble-graph")
        ])
    ]),

    dcc.Store(id='dataset'),
    dcc.Store(id='slider-marks')
], style={"max-width": "100%", "paddingTop": "12px"})


@app.callback(
    Output("dataset", "data"),
    Output("slider-bubble", "min"),
    Output("slider-bubble", "max"),
    Output("slider-bubble", "marks"),
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

    # Clean the slider marks to only show one per half year
    slider_marks_cleaned = {}
    for key in [*slider_marks]:
        if key != 0 and key != len(unique_months)-1:
            if slider_marks[key].endswith("01") or slider_marks[key].endswith("07"):
                slider_marks_cleaned[key] = slider_marks[key]
        else:
            slider_marks_cleaned[key] = slider_marks[key]

    return df.to_json(), 0, len(unique_months)-1, slider_marks_cleaned, json.dumps(slider_marks)


def get_top_songs_range(df, start_range=None, end_range=None, range_column=None, top=5):

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

    tracks = df[['master_metadata_track_name', 'master_metadata_album_artist_name', 'master_metadata_album_album_name', 'spotify_track_uri']].copy()
    counts = df['spotify_track_uri'].value_counts(sort=False).reset_index()
    tracks = tracks.drop_duplicates().reset_index(drop=True)
    tracks['count'] = counts['spotify_track_uri']
    tracks.sort_values(by=['count'], ascending=False, inplace=True)
    tracks.reset_index(drop=True, inplace=True)
    return tracks.head(top)


@app.callback(
    Output("top-tracks", "children"),
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

        return convert_to_top_tracks_list(top_tracks)

    # When the user has resized the graph
    if "xaxis.range[0]" in graph_events:
        app.logger.info(graph_events)
        top_tracks = get_top_songs_range(df, graph_events["xaxis.range[0]"], graph_events["xaxis.range[1]"], timespan)

        return convert_to_top_tracks_list(top_tracks)


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


@app.callback(
    Output("bubble-graph", "figure"),
    Input("dataset", "data"),
    Input("slider-bubble", "value"),
    Input("slider-marks", "data")
)
def update_bubble_graph(data, month, slider_marks):

    slider_marks = json.loads(slider_marks)
    month = 0 if month is None else month

    df = pd.read_json(data)
    top = 9

    data = get_top_artists(df, slider_marks[str(month)], top)

    circles = circlify.circlify(
        data['counts'], 
        show_enclosure=False, 
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )

    child_circle_groups = []
    for i in range(len(data['counts'])):
        child_circle_groups.append(circlify.circlify(
            data['counts'], 
            show_enclosure=False, 
            target_enclosure=circlify.Circle(x=circles[i].x, y=circles[i].y, r=circles[i].r)
        ))

    fig = go.Figure()

    # Set axes properties
    fig.update_xaxes(
        range=[-1.05, 1.05], # making slightly wider axes than -1 to 1 so no edge of circles cut-off
        showticklabels=False,
        showgrid=False,
        zeroline=False
    )

    fig.update_yaxes(
        range=[-1.05, 1.05],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    # Add parent circles
    for idx, circle in enumerate(circles):
        x, y, r = circle
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=x-r, y0=y-r,
            x1=x+r, y1=y+r,
            fillcolor=data['colors'][top-1-idx],
            line_width=0,
        )

        # TODO: Find a way to center the annotation in the circle
        # --> Move it down by 8px
        fig.add_annotation(
            x=x, y=y,
            text=data['tracks'][top-1-idx],
            font=dict(
                color="#000000"
            ),
            showarrow=False,
            yshift=10
        )

    # Set figure size
    fig.update_layout(width=800, height=800, plot_bgcolor="white")

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True) # Running in debug mode makes the server reload automatically on changes

