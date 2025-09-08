#! ./.venv/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import re

SOC_MIN = 0.01
SOC_MAX = 1.00
SOC_STEP = 0.001
TEMP_REF = 25.0
FIT_POINTS = 1000


def plot_soc_ocv(df, *params):
    a, b, c, d, e, f, alpha = params
    temps = sorted(df["temp"].unique())
    soc_fit = np.linspace(0, SOC_MAX, FIT_POINTS)

    fig, ax = plt.subplots(figsize=(16, 9))

    viridis = plt.get_cmap("viridis")
    colors = [viridis(i) for i in np.linspace(0, 1, len(temps))]

    for i, T in enumerate(temps):
        # Filter data for this temperature
        df_temp = df[df["temp"] == T]

        # Scatter: measured points
        ax.scatter(
            df_temp["soc"], df_temp["ocv"], color=colors[i], s=20, label=f"{T}°C data"
        )

        ocv_fit = ocv_model([soc_fit, T], a, b, c, d, e, f, alpha)
        ax.plot(soc_fit, ocv_fit, color=colors[i], linestyle="--", label=f"{T}°C fit")

    ax.set_xlabel("SOC")
    ax.set_ylabel("OCV [V]")
    ax.set_title("Global OCV(SOC, T) fit")
    ax.grid(True, alpha=0.5, linestyle="--")
    ax.legend()
    plt.show()


def r_alpha(alpha, value, r_prev):
    return alpha * value + (1 - alpha) * r_prev


def detect_edges(array):
    alpha_fast = 0.4
    alpha_slow = 0.15
    delta_threshold = 0.5

    r_fast = array[0]
    r_slow = array[0]
    delta = r_fast - r_slow

    edges = []
    flag = False
    for i in range(1, len(array)):
        r_fast = r_alpha(alpha_fast, array[i], r_fast)
        r_slow = r_alpha(alpha_slow, array[i], r_slow)
        delta = r_fast - r_slow

        if abs(delta) > delta_threshold:
            # edges.append(i)
            if not flag:
                edges.append(i)
            flag = True
        else:
            flag = False

    return edges


def ocv_model(inputs, a, b, c, d, e, f, alpha):
    soc, temp = inputs
    base = a + b * np.log(soc + c) + d * soc + np.exp(e * (soc - f))
    temp_term = alpha * (temp - TEMP_REF)
    return base + temp_term


def analyze_dataset(file: str, temp: float):
    # get data
    df = pd.read_csv(
        file,
        engine="python",
        skiprows=49,
        skipfooter=1,
        encoding="latin1",
        names=["time", "voltage", "current", "t_env", "t_cell"],
    )

    # unpack data into np.array
    t = df["time"].to_numpy()
    v = df["voltage"].to_numpy()
    c = df["current"].to_numpy()

    t_unique = np.unique(t)
    sampling_freq = np.mean(1 / (t_unique[1:] - t_unique[:-1]))

    edges = detect_edges(c)

    # filter voltages and currents to eliminate noise
    samples_num = 5
    v_f = np.sum(
        np.lib.stride_tricks.sliding_window_view(v, 2 * samples_num + 1), axis=1
    ) / (2 * samples_num + 1)
    c_f = np.sum(
        np.lib.stride_tricks.sliding_window_view(c, 2 * samples_num + 1), axis=1
    ) / (2 * samples_num + 1)

    # get ocv values by averaging over a 10 second period before each edge
    ocv_avg_time = 10
    chunk_size = int(ocv_avg_time * sampling_freq)
    skip_factor = 6
    ocv_values = []
    for edge in edges[::skip_factor]:
        start = max(edge - chunk_size, 0)
        ocv_values.append(np.mean(v_f[start:edge]))
    ocv_values = np.array(ocv_values[:-chunk_size:-1])

    soc = np.linspace(SOC_MIN, SOC_MAX, len(ocv_values))

    return ocv_values, soc


def format_array_c(name, array, width):
    lines = [f"static const float {name}[{len(array)}] = {{"]
    for i in range(0, len(array), width):
        chunk = ", ".join(f"{v:.6f}f" for v in array[i : i + width])
        lines.append("    " + chunk + ",")
    lines[-1] = lines[-1].rstrip(",")  # Remove last comma
    lines.append("};\n")
    return "\n".join(lines)


def data2c(*params):
    header = """/*
* ocv_lut.h
*
* Auto-generated LUT for battery OCV vs SoC characteristic
*
* DO NOT MODIFY!
*
*/\n
"""
    lut_soc = np.arange(SOC_STEP, SOC_MAX + SOC_STEP, SOC_STEP, np.float64)
    lut_ocv = ocv_model([lut_soc, TEMP_REF], *params)
    lut_soc *= 100.0

    len_arr = len(lut_soc)

    width = 10
    soc_table_name = "soc_table"
    ocv_table_name = "ocv_table"
    lut_soc_c_fmt = format_array_c(soc_table_name, lut_soc, width)
    lut_ocv_c_fmt = format_array_c(ocv_table_name, lut_ocv, width)

    lookup_func_name = "ocv_lookup"

    define_name = "TABLE_LENGTH"

    with open("ocv_lut.h", "w") as f:
        f.write(header)
        f.write("#ifndef OCV_LUT_H\n")
        f.write("#define OCV_LUT_H\n\n")
        f.write(f"#define {define_name} {len_arr}\n\n")
        f.write(lut_ocv_c_fmt)
        f.write("\n")
        f.write(lut_soc_c_fmt)
        f.write(
            f"""
/* linear interpolation on LUT x and y */
static float {lookup_func_name}(float ocv) {{
    if (ocv <= {ocv_table_name}[0]) return {soc_table_name}[0];
    if (ocv >= {ocv_table_name}[{define_name} - 1]) return {soc_table_name}[{define_name} - 1];

    for (int i = 0; i < {define_name} - 1; ++i) {{
        if (ocv >= {ocv_table_name}[i] && ocv <= {ocv_table_name}[i + 1]) {{
            float ocv_0 = {ocv_table_name}[i];
            float ocv_1 = {ocv_table_name}[i + 1];
            float soc_0 = {soc_table_name}[i];
            float soc_1 = {soc_table_name}[i + 1];
            float t = (ocv - ocv_0) / (ocv_1 - ocv_0);
            return soc_0 + t * (soc_1 - soc_0);
        }}
    }}

    /* unreachable */
    return 0.0f;
}}
"""
        )
        f.write("\n#endif /* OCV_LUT */\n")


def main():
    data = []

    for file in glob.glob("data/**/*.csv", recursive=True):
        temp_match = re.search(r"(\d+)deg", file)
        temp = int(temp_match.group(1)) if temp_match else 25
        print(f"Processing {file} file")
        ocv_values, soc = analyze_dataset(file, temp)
        for o, s in zip(ocv_values, soc):
            data.append({"soc": s, "ocv": o, "temp": temp})

    df = pd.DataFrame(data)

    soc_all = np.array(df.get("soc", 0), np.float64)
    ocv_all = np.array(df.get("ocv", 0), np.float64)
    temp_all = np.array(df.get("temp", 0), np.float64)

    xdata = np.vstack([soc_all, temp_all])
    ydata = ocv_all

    # Initial guess for the parameters [a, b, c, d, e, f, alpha]
    initial_guess = [3.5, 0.1, 0.00, -0.2, 400, 1, 0]
    popt, pcov = curve_fit(ocv_model, xdata, ydata, p0=initial_guess, maxfev=20000)

    # Extract fitted parameters
    a, b, c, d, e, f, alpha = popt
    print("Fitted parameters:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}")
    print(f"d = {d}")
    print(f"e = {e}")
    print(f"f = {f}")
    print(f"alpha = {alpha}")

    plot_soc_ocv(df, a, b, c, d, e, f, alpha)

    data2c(*popt)


if __name__ == "__main__":
    main()
