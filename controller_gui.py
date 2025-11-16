import threading
import time
from dataclasses import dataclass, asdict

import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

import so101

# =========================
# Shared twist data model
# =========================
twist_lock = threading.Lock()


# =========================
# Robot control stub
# =========================
robot = so101.SO101()

def control_loop():
    """
    Background loop that runs in parallel with the Dash web server.
    It continuously sends either the requested twist or zero twist
    to hold position.
    """
    rate_hz = 20.0  # send commands at 20 Hz
    dt = 1.0 / rate_hz

    zero_cmd = TwistCommand()  # all zeros, active=False by default

    while True:
        with twist_lock:
            # Make a copy so we don't hold the lock during send
            current = TwistCommand(**asdict(twist_cmd))

        current_pos = robot.read_current_position()

        if current.active:
            # Send the requested twist
            target_twist = current
        else:
            # Explicitly send zero to hold position
            target_twist = zero_cmd

        time.sleep(dt)


# =========================
# Dash app
# =========================

app: Dash = dash.Dash(__name__)

app.layout = html.Div(
    style={"maxWidth": "500px", "margin": "2rem auto", "fontFamily": "sans-serif"},
    children=[
        html.H2("Robot Arm Twist Controller"),

        html.Div(
            style={"marginBottom": "1rem"},
            children=[
                html.H4("Linear velocity (cm/s)"),
                html.Div(
                    style={"display": "flex", "gap": "0.5rem"},
                    children=[
                        html.Div([
                            html.Label("vₓ"),
                            dcc.Input(id="vx-input", type="number", value=0.0, step=1),
                        ]),
                        html.Div([
                            html.Label("vᵧ"),
                            dcc.Input(id="vy-input", type="number", value=0.0, step=1),
                        ]),
                        html.Div([
                            html.Label("v𝓏"),
                            dcc.Input(id="vz-input", type="number", value=0.0, step=1),
                        ]),
                    ],
                ),
            ],
        ),

        html.Div(
            style={"marginBottom": "1rem"},
            children=[
                html.H4("Angular velocity (deg/s, step 5)"),
                html.Div(
                    style={"display": "flex", "gap": "0.5rem"},
                    children=[
                        html.Div([
                            html.Label("ωₓ"),
                            dcc.Input(id="wx-input", type="number", value=0.0, step=5),
                        ]),
                        html.Div([
                            html.Label("ωᵧ"),
                            dcc.Input(id="wy-input", type="number", value=0.0, step=5),
                        ]),
                        html.Div([
                            html.Label("ω𝓏"),
                            dcc.Input(id="wz-input", type="number", value=0.0, step=5),
                        ]),
                    ],
                ),
            ],
        ),

        html.Button(
            "Execute twist (currently: HOLD)",
            id="execute-button",
            n_clicks=0,
            style={"padding": "0.5rem 1rem", "marginBottom": "1rem"},
        ),

        html.Div(id="status-text", style={"marginTop": "0.5rem", "fontStyle": "italic"}),
    ],
)


# =========================
# Dash callback
# =========================

@app.callback(
    Output("execute-button", "children"),
    Output("status-text", "children"),
    Input("execute-button", "n_clicks"),
    State("vx-input", "value"),
    State("vy-input", "value"),
    State("vz-input", "value"),
    State("wx-input", "value"),
    State("wy-input", "value"),
    State("wz-input", "value"),
)
def update_twist(
    n_clicks,
    vx, vy, vz,
    wx, wy, wz,
):
    """
    - Button works as a toggle:
        * Even clicks -> inactive (zero twist / hold)
        * Odd  clicks -> active (send twist from fields)
    - When inactive, the background loop sends zero twist.
    """
    if not n_clicks:
        active = False
    else:
        active = (n_clicks % 2 == 1)

    # Default to 0.0 if any field is None
    vx = vx or 0.0
    vy = vy or 0.0
    vz = vz or 0.0
    wx = wx or 0.0
    wy = wy or 0.0
    wz = wz or 0.0

    with twist_lock:
        twist_cmd.vx = vx
        twist_cmd.vy = vy
        twist_cmd.vz = vz
        twist_cmd.wx = wx
        twist_cmd.wy = wy
        twist_cmd.wz = wz
        twist_cmd.active = active

    if active:
        button_label = "Execute twist (currently: ACTIVE)"
        status = (
            f"Sending twist: "
            f"v = [{vx:.2f}, {vy:.2f}, {vz:.2f}] cm/s, "
            f"ω = [{wx:.2f}, {wy:.2f}, {wz:.2f}] deg/s"
        )
    else:
        button_label = "Execute twist (currently: HOLD)"
        status = "Holding position (zero twist commanded)."

    return button_label, status


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    # Start the control loop in the background
    t = threading.Thread(target=control_loop, daemon=True)
    t.start()

    # Start the Dash app
    app.run(debug=True)
