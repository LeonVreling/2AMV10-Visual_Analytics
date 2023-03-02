# app.py

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import spotipy
from spotipy.oauth2 import SpotifyOAuth


# Authenticate using the Spotify server
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="fc126aaa02334aae871ae10bdba19854",
                                               client_secret="77feae55e0b949788ea1e3de052e4230",
                                               redirect_uri="http://localhost:8085/callback/",
                                               scope="user-library-read, user-top-read"))


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])


tracks = sp.current_user_saved_tracks()
track_options = [{"label": track["track"]["name"], "value": track["track"]["id"]} for track in tracks["items"]]


top_tracks = sp.current_user_top_tracks(limit=10, time_range='short_term')
top_tracks = [{"title": track["name"], "artist": track["artists"][0]["name"], "id": track["id"]} for track in top_tracks["items"]]
app.logger.info(top_tracks)

top_tracks_features = sp.audio_features(tracks=[track["id"] for track in top_tracks])
app.logger.info(top_tracks_features)

titles = [track["title"] for track in top_tracks]
artists = [track["artist"] for track in top_tracks]
tempos = [track["tempo"] for track in top_tracks_features]
durations = [track["duration_ms"] / 1000 for track in top_tracks_features]

df = pd.DataFrame({
    "Title": titles,
    "Artist": artists,
    "Tempo": tempos,
    "Duration": durations
})

fig = px.scatter(df, x="Tempo", y="Duration",
                 custom_data=["Title", "Artist"],
                 labels={"Tempo": "Tempo (bpm)", "Duration": "Duration (s)"},
                 title="Top 10 songs of this month")

fig.update_traces(
    hovertemplate="<br>".join([
        "Title: %{customdata[0]}",
        "Artist: %{customdata[1]}"
    ])
)



app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Analyzing Spotfiy data", style={'textAlign': 'center'})
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.P("A simple web dashboard for analyzing Spotify data using various visual analytics and machine learning techniques."),
            html.I(className="fa-solid fa-people-group"),
            html.Span(" Marlou Gielen, Elian Klaassen and Leon Vreling"),
            html.Br(),
            html.I(className="fa-solid fa-graduation-cap"),
            html.Span(" Eindhoven University of Technology, 2AMV10 Visual Analytics")
        ], width=6),

        dbc.Col([
            html.Img(src=app.get_asset_url("spotify_logo.png"), style={'width': '100%'})
        ], width=6)
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id="track-dropdown",
                options=track_options,
                value=track_options[0]["value"]
            )
        ], width=6),

        dbc.Col([
            html.Div(id="track-info")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Ol([html.Li(children=[track["artist"], " - ", track["title"]]) for track in top_tracks])
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='example-graph',
                figure=fig
            )
        ])
    ])


])


@app.callback(
    Output("track-info", "children"),
    [Input("track-dropdown", "value")]
)
def update_track_info(track_id):
    track = sp.track(track_id)
    return html.Div([
        html.H2(track["name"]),
        html.P(track["artists"][0]["name"])
    ])


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True) # Running in debug mode makes the server reload automatically on changes

