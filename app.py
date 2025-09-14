import analysis
import plotly.graph_objects as go
import re
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("dark_background")


@st.cache_data
def load_data(file: str):
    return analysis.load_data(file)


st.title("HPPC Data Analysis")

st.markdown("""
Analysis of a battery cell for an EV vehicle prototype,
based on [this paper](https://www.mdpi.com/1996-1073/16/17/6239)
""")

file_paths: list[str] = analysis.get_data_files()

selected_file: str = st.selectbox("Select file", file_paths)

temp_match = re.search(r"(\d+)deg", selected_file)
TEMP = int(temp_match.group(1)) if temp_match else 25

# Extract C-rate from filename
crate_match = re.search(r"(\d+)C", selected_file)
CRATE = int(crate_match.group(1)) if crate_match else 1

data_load_state = st.text("Loading data...")
df = load_data(selected_file)
data_load_state.text("")
st.dataframe(df)

t = np.array(df.get("time", 0))
v = np.array(df.get("voltage", 0))
c = np.array(df.get("current", 0))


st.subheader("Raw Data")
hours = t[np.size(t) - 1] / 3600
raw_data_body = f"Data aquired during the span of ~{round(hours):.0f} hours"
st.write(raw_data_body)

fig, ax = analysis.plot_setup(
    f"Voltage readings @ {CRATE}C and {TEMP}°C",
    "Time (s)",
    "Voltage (V)",
)
ax.plot(t, v, color="y", label="HPPC test voltage readings")
ax.legend()
st.pyplot(fig)

fig, ax = analysis.plot_setup(
    f"Current readings @ {CRATE}C and {TEMP}°C",
    "Time (s)",
    "Current (A)",
)
ax.plot(t, c, color="y", label="HPPC test voltage readings")
ax.legend()
st.pyplot(fig)

st.subheader("Edge detection")
t_unique = np.unique(t)
sampling_freq = np.mean(1 / (t_unique[1:] - t_unique[:-1]))


# filter voltages and currents to eliminate noise
samples_num = 5
t_f = np.sum(
    np.lib.stride_tricks.sliding_window_view(t, 2 * samples_num + 1), axis=1
) / (2 * samples_num + 1)
v_f = np.sum(
    np.lib.stride_tricks.sliding_window_view(v, 2 * samples_num + 1), axis=1
) / (2 * samples_num + 1)
c_f = np.sum(
    np.lib.stride_tricks.sliding_window_view(c, 2 * samples_num + 1), axis=1
) / (2 * samples_num + 1)

edges = analysis.detect_edges(c_f)

t_ds = t_f[::10]
c_ds = c_f[::10]
v_ds = v_f[::10]



st.write("""
Detecting the edges in the current draw enables us to find when
the cell is being discharged and construct the OCV vs SoC curve
calculating the amount of charge drawn by the cell during discharge
and relating it to the nominal cell capacity.
""")


fig = go.Figure()
fig.add_trace(go.Scatter(x=t_ds, y=c_ds, mode="lines", name="Current Filtered"))
vlines_color = "yellow"
vlines_style = "dot"
for vline in t_f[edges]:
    fig.add_vline(
        x=vline,
        line_width=1,
        line_dash=vlines_style,
        line_color=vlines_color,
    )
fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color=vlines_color, dash=vlines_style),
        name="Detected Edges",
    )
)
fig.update_layout(
    title=dict(text="Edges in current draw", font=dict(size=18)),
    xaxis=dict(title="Time (s)", title_font=dict(size=18), tickfont=dict(size=14)),
    yaxis=dict(title="Current (A)", title_font=dict(size=18), tickfont=dict(size=14)),
    legend=dict(font=dict(size=14)),
    height=600,
)
st.plotly_chart(fig, use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=t_ds, y=v_ds, mode="lines", name="Tension Filtered"))
vlines_color = "yellow"
vlines_style = "dot"
for vline in t_f[edges]:
    fig.add_vline(
        x=vline,
        line_width=1,
        line_dash=vlines_style,
        line_color=vlines_color,
    )
fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color=vlines_color, dash=vlines_style),
        name="Detected Edges",
    )
)
fig.update_layout(
    title=dict(text="Edges in voltages", font=dict(size=18)),
    xaxis=dict(title="Time (s)", title_font=dict(size=18), tickfont=dict(size=14)),
    yaxis=dict(title="Voltage (V)", title_font=dict(size=18), tickfont=dict(size=14)),
    legend=dict(font=dict(size=14)),
    height=600,
)
st.plotly_chart(fig, use_container_width=True)



# analysis
ocv_values, soc = analysis.analyze_dataset(df)
data_ocv = pd.DataFrame(
    {
        "temp": TEMP,
        "soc": soc,
        "ocv": ocv_values,
    }
)
xdata: np.ndarray = np.vstack([soc, data_ocv["temp"]])
ydata: np.ndarray = ocv_values

# fit
initial_guess: list[float] = [3.5, 0.1, 0.00, -0.2, 400, 1, 0]
popt = initial_guess
popt, _ = analysis.fit_ocv_model(xdata, ydata, initial_guess)
a, b, c, d, e, f, alpha = popt

# display parameters
columns = ["a", "b", "c", "d", "e", "f", "alpha"]
values = [[a, b, c, d, e, f, alpha]]
parameters = pd.DataFrame(values, columns=columns)

# plot ocv vs soc
fig, ax = analysis.plot_soc_ocv(
    soc,
    ocv_values,
    f"Voltage readings @ {CRATE}C and {TEMP}°C",
    "SoC (%)",
    "OCV (V)",
    *popt,
)
st.subheader("OCV vs SoC curve parameter estimation")
st.write("""
Perform the fit with a **log-linear exponential (LEE)** function:
""")
st.latex(r"""
a + b \cdot \log(\text{SoC} + c) + d \cdot \text{SoC} + \exp\left[e (\text{SoC} - f)\right]
""")
st.pyplot(fig)
st.write("Estimated Parameters:")
st.markdown(parameters.to_html(index=False), unsafe_allow_html=True)

st.subheader("Battery model parameter estimation")
edges_sensible = analysis.detect_edges(c_f, alpha_slow=0.6, alpha_fast=0.30, delta_threshold=0.1)
start = edges_sensible[0]
end = edges_sensible[1]
t_s = t_f[start:end] - t_f[start]
v_s = v_f[start:end]
c_s = c_f[start:end]
ocv_start_values = np.full_like(t_s, 4.2)
battery_inputs = np.vstack([t_s, -c_s, ocv_start_values])
battery_params = analysis.battery_parameter_estimation(df)
r0, r1, r2, t1, t2 = battery_params
# compute capacities
c1 = t1 / r1 if r1 > 0 else np.nan
c2 = t2 / r2 if r2 > 0 else np.nan

model_bat_voltage = analysis.battery_model(battery_inputs, *battery_params)
fig, ax = analysis.plot_setup("", "", "")
ax.scatter(t_s, v_s, label="Data", color="y")
ax.plot(t_s, model_bat_voltage, label="Fitted model", c="r", ls="--")
ax.legend()
st.pyplot(fig)

filename = "ocv_lut.h"
st.subheader("Example output file: " + filename)
st.write("""
As this data is to be used in an embedded environment,
we can generate a table to embed in a .c/.h file.
""")
num_values = st.number_input("Table length", min_value=2, step=1, value=100)
c_code = analysis.model2c(num_values, 10, *popt)

st.code(c_code, "c", line_numbers=True)

# Download button
st.download_button(
    label=f"Download *{filename}*", data=c_code, file_name=filename, mime="text/plain"
)
