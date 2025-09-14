from typing import List, Sequence, Tuple, Any
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.optimize import curve_fit, Bounds
import glob
import re

# TODO(siegmone): generate model.h file with battery parameter estimation and lut

SOC_MIN = 0.01
SOC_MAX = 1.00
SOC_STEP = 0.001
TEMP_REF = 25.0
FIT_POINTS = 1000

FONT = "DejaVu Sans"
FONTSIZE = 10


def plot_setup(title: str, xlabel: str, ylabel: str) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.grid(True, alpha=0.4, linestyle="--")
    ax.set_title(title, fontsize=FONTSIZE, fontfamily=FONT)

    ax.set_xlabel(xlabel, fontsize=FONTSIZE, fontfamily=FONT)

    ax.set_ylabel(ylabel, fontsize=FONTSIZE, fontfamily=FONT)

    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    return fig, ax


def plot_soc_ocv(
    soc: np.ndarray, ocv: np.ndarray, title: str, xlabel: str, ylabel: str, *params
) -> Tuple[Figure, Axes]:
    fig, ax = plot_setup(title, xlabel, ylabel)

    soc_fit = np.linspace(0, SOC_MAX, FIT_POINTS)
    ocv_fit = ocv_model([soc_fit, TEMP_REF], *params)

    soc *= 100
    soc_fit *= 100

    ax.set_xlim(0, 100)

    ax.scatter(soc, ocv, color="y", s=50, marker="X", label="SoC vs OCV data")
    ax.plot(soc_fit, ocv_fit, color="b", linestyle="-.", label="Fitted curve")
    ax.legend()

    return fig, ax


def r_alpha(alpha: float, value: float, r_prev: float) -> float:
    return alpha * value + (1 - alpha) * r_prev


def detect_edges(
    array: np.ndarray | Sequence[float],
    alpha_fast: float = 0.4,
    alpha_slow: float = 0.15,
    delta_threshold: float = 0.5,
) -> List[int]:
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


def ocv_model(inputs: Sequence[np.ndarray | float], *params: float) -> np.ndarray:
    a, b, c, d, e, f, alpha = params
    soc, temp = inputs
    base = a + b * np.log(soc + c) + d * soc + np.exp(e * (soc - f))
    temp_term = alpha * (temp - TEMP_REF)
    _ = temp_term  # NOTE: don't use, it breaks the fit
    return base


def battery_model(inputs: Sequence[np.ndarray | float], *params: float) -> np.ndarray:
    time, current, ocv = inputs
    r0, r1, r2, t1, t2 = params
    return (
        ocv
        - current * r0
        - current * r1 * (1 - np.exp(-time / t1))
        - current * r2 * (1 - np.exp(-time / t2))
    )


def parameter_estimation(t: np.ndarray, v: np.ndarray, c: np.ndarray, ocv_value: float, start: int, end: int):
    t_s = t[start:end] - t[start]
    v_s = v[start:end]
    c_s = c[start:end]

    ocv_values = np.full_like(t_s, ocv_value)
    xdata = np.vstack([t_s, -c_s, ocv_values])
    ydata = v_s

    lower = [
        0.0,
        0.0,
        0.0,
        1e-6,
        1e-6
    ]
    upper = [
        0.5,
        0.5,
        0.5,
        1e5,
        1e6
    ]
    bounds = Bounds(lower, upper)

    p0 = [0.01, 0.01, 0.01, 1.0, 100.0]

    popt, pcov = curve_fit(
        battery_model,
        xdata=xdata,
        ydata=ydata,
        p0=p0,
        bounds=bounds,
        max_nfev=20000,
    )

    r0, r1, r2, t1, t2 = popt
    _ = np.sqrt(np.diag(pcov))

    return popt


def battery_parameter_estimation(df: pd.DataFrame) -> List[float]:
    t, v, c = unpack_dataframe(df)

    t_unique = np.unique(t)
    sampling_freq = np.mean(1 / (t_unique[1:] - t_unique[:-1]))
    ocv_avg_time = 10
    chunk_size = int(ocv_avg_time * sampling_freq)
    skip_factor = 6

    # filter voltages and currents to eliminate noise
    samples_num = 5
    t_f = t
    v_f = np.sum(
        np.lib.stride_tricks.sliding_window_view(v, 2 * samples_num + 1), axis=1
    ) / (2 * samples_num + 1)
    c_f = np.sum(
        np.lib.stride_tricks.sliding_window_view(c, 2 * samples_num + 1), axis=1
    ) / (2 * samples_num + 1)

    edges = detect_edges(c_f, alpha_slow=0.6, alpha_fast=0.30, delta_threshold=0.1)

    params_all = []
    for i in range(len(edges) // skip_factor):
        for j in range(2):
            start = edges[i * skip_factor + 2 * j]
            end = edges[i * skip_factor + 2 * j + 1]
            ocv_value = np.mean(v_f[start - chunk_size : start])
            parameters = parameter_estimation(t_f, v_f, c_f, ocv_value, start, end)
            params_all.append(parameters)

    params = np.mean(params_all, axis=0)

    return params



def unpack_dataframe(df: pd.DataFrame):
    t = np.array(df.get("time", 0))
    v = np.array(df.get("voltage", 0))
    c = np.array(df.get("current", 0))
    return t, v, c


def analyze_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # unpack data into np.array
    t, v, c = unpack_dataframe(df)

    t_unique = np.unique(t)
    sampling_freq = np.mean(1 / (t_unique[1:] - t_unique[:-1]))

    # filter voltages and currents to eliminate noise
    samples_num = 5
    v_f = np.sum(
        np.lib.stride_tricks.sliding_window_view(v, 2 * samples_num + 1), axis=1
    ) / (2 * samples_num + 1)
    c_f = np.sum(
        np.lib.stride_tricks.sliding_window_view(c, 2 * samples_num + 1), axis=1
    ) / (2 * samples_num + 1)

    edges = detect_edges(c_f)

    # get ocv values by averaging over a 10 second period before each edge
    ocv_avg_time = 10
    chunk_size = int(ocv_avg_time * sampling_freq)
    skip_factor = 6
    ocv_values: List[float] = []
    for edge in edges[::skip_factor]:
        start = max(edge - chunk_size, 0)
        ocv_values.append(np.mean(v_f[start:edge]))
    ocv_values_arr = np.array(ocv_values[:-chunk_size:-1])

    soc: np.ndarray = np.linspace(SOC_MIN, SOC_MAX, len(ocv_values_arr))

    return ocv_values_arr, soc


def format_array_c(
    name: str,
    array: np.ndarray | Sequence[float],
    width: int,
) -> str:
    lines = [f"static const float {name}[{len(array)}] = {{"]
    for i in range(0, len(array), width):
        chunk = ""
        for v in array[i : i + width]:
            if int(v) == v:
                chunk = ", ".join(f"{v:.1f}f" for v in array[i : i + width])
            else:
                chunk = ", ".join(f"{v:.3f}f" for v in array[i : i + width])
        lines.append("    " + chunk + ",")
    lines.append("};\n")
    return "\n".join(lines)


def model2c(num_values: int, col_width: int, *params: float) -> str:
    file_body = ""
    header = """/*
* ocv_lut.h
*
* Auto-generated LUT for battery OCV vs SoC characteristic
*
* DO NOT MODIFY!
*
*/\n
"""
    lut_soc: np.ndarray = np.linspace(SOC_MIN, SOC_MAX, num_values)
    lut_ocv: np.ndarray = ocv_model([lut_soc, TEMP_REF], *params)
    lut_soc *= 100.0

    len_arr: int = len(lut_soc)
    define_name = "TABLE_LENGTH"

    soc_table_name = "soc_table"
    ocv_table_name = "ocv_table"
    lut_soc_c_fmt: str = format_array_c(soc_table_name, lut_soc, col_width)
    lut_ocv_c_fmt: str = format_array_c(ocv_table_name, lut_ocv, col_width)

    soc_lookup_func_name = "soc_lookup"
    soc_lookup_func = f"""
\n\n/* performs linear interpolation on ocv vs soc curve values */
static float {soc_lookup_func_name}(float ocv) {{
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

    ocv_lookup_func_name = "ocv_lookup"
    ocv_lookup_func = f"""
\n\n/* performs linear interpolation on ocv vs soc curve values */
static float {ocv_lookup_func_name}(float soc) {{
    if (soc <= {ocv_table_name}[0]) return {soc_table_name}[0];
    if (soc >= {ocv_table_name}[{define_name} - 1]) return {soc_table_name}[{define_name} - 1];

    for (int i = 0; i < {define_name} - 1; ++i) {{
        if (soc >= {ocv_table_name}[i] && soc <= {ocv_table_name}[i + 1]) {{
            float ocv_0 = {ocv_table_name}[i];
            float ocv_1 = {ocv_table_name}[i + 1];
            float soc_0 = {soc_table_name}[i];
            float soc_1 = {soc_table_name}[i + 1];
            float t = (soc - soc_0) / (soc_1 - soc_0);
            return ocv_0 + t * (ocv_1 - ocv_0);
        }}
    }}

    /* unreachable */
    return 0.0f;
}}
"""


    file_body += header
    file_body += "#ifndef OCV_LUT_H\n"
    file_body += "#define OCV_LUT_H\n\n"
    file_body += f"#define {define_name} {len_arr}\n\n"
    file_body += lut_ocv_c_fmt
    file_body += "\n"
    file_body += lut_soc_c_fmt
    file_body += soc_lookup_func
    file_body += ocv_lookup_func
    file_body += "\n#endif /* OCV_LUT */\n"

    return file_body


def load_data(file: str) -> pd.DataFrame:
    df = pd.read_csv(
        file,
        engine="python",
        skiprows=49,
        skipfooter=1,
        encoding="latin1",
        names=["time", "voltage", "current", "t_env", "t_cell"],
    )
    df["time"] -= df["time"].iloc[0]
    return df


def get_data_files() -> List[Any]:
    return glob.glob("data/**/*.csv", recursive=True)


def fit_ocv_model(xdata: np.ndarray, ydata: np.ndarray, initial_guess: List[float]):
    popt, pcov = curve_fit(ocv_model, xdata, ydata, p0=initial_guess, maxfev=20000)
    return popt, pcov


def main():
    dataframe_list = []
    data: List[dict[str, float | int]] = []
    file_list = get_data_files()

    for file in file_list:
        temp_match = re.search(r"(\d+)deg", file)
        temp = int(temp_match.group(1)) if temp_match else 25
        df = load_data(file)
        dataframe_list.append(df)
        ocv_values, soc = analyze_dataset(df)
        for o, s in zip(ocv_values, soc):
            data.append({"soc": s, "ocv": o, "temp": temp})
        break

    df_all = pd.DataFrame(data)

    soc_all: np.ndarray = np.array(df_all.get("soc", 0), np.float64)
    ocv_all: np.ndarray = np.array(df_all.get("ocv", 0), np.float64)
    temp_all: np.ndarray = np.array(df_all.get("temp", 0), np.float64)

    xdata: np.ndarray = np.vstack([soc_all, temp_all])
    ydata: np.ndarray = ocv_all

    initial_guess: list[float] = [3.5, 0.1, 0.00, -0.2, 400, 1, 0]
    soc_ocv_parameters, _ = fit_ocv_model(xdata, ydata, initial_guess)

    # Initial guess for the parameters [a, b, c, d, e, f, alpha]
    # Extract fitted parameters
    a, b, c, d, e, f, alpha = soc_ocv_parameters
    print("Fitted parameters:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}")
    print(f"d = {d}")
    print(f"e = {e}")
    print(f"f = {f}")
    print(f"alpha = {alpha}")

    fig, ax = plot_soc_ocv(
        soc_all,
        ocv_all,
        "Voltage readings",
        "SoC (%)",
        "OCV (V)",
        *soc_ocv_parameters,
    )
    plt.show()

    battery_parameters = battery_parameter_estimation(dataframe_list[0])
    r0, r1, r2, c1, c2 = battery_parameters

    output_file_text = model2c(1000, 10, *soc_ocv_parameters)

    output_file_text += f"""
\n\n/*
* Estimated battery parameters
* R0 = {r0:.6f}, R1 = {r1:.6f}, R2 = {r2:.6f}, C1 = {c1:.6f}, C2 = {c2:.6f}
*/
    """

    with open("ocv_lut.h", "w") as f:
        f.write(output_file_text)


if __name__ == "__main__":
    main()
