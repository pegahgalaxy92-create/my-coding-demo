import numpy as np
import pandas as pd
from EV_CS_Model_Module import EV_CS_Solve
from MCDM import mcdm_rank_analysis

number_payoff=6

# محاسبه r2 و آرایه‌های e2
try:
    result_max = EV_CS_Solve(index_obj=2, type_opt=1,gap=0.05)[0]
    result_min = EV_CS_Solve(index_obj=2, type_opt=-1)[0]

    if not result_max or not result_min:
        raise ValueError("EV_CS_Solve for f2 returned empty result.")

    max_f2 = result_max
    min_f2 = result_min
    r2 = round(max_f2 - min_f2)
    e2_arr = np.linspace(min_f2, max_f2, num=number_payoff)

    print(f"r2 = {r2}")

except Exception as e:
    print(f"Error in computing r2: {e}")
    e2_arr = []
    r2 = None

try:
    result_max = EV_CS_Solve(index_obj=3, type_opt=1,gap=0.05)[0]
    result_min = EV_CS_Solve(index_obj=3, type_opt=-1)[0]

    if result_max is None or result_min is None:
        raise ValueError("EV_CS_Solve returned None.")

    max_f3 = result_max  # اگر عدد هستند، مستقیم ذخیره می‌کنیم
    min_f3 = result_min

    r3 = round(max_f3 - min_f3)
    e3_arr = np.linspace(min_f3, max_f3, num=number_payoff)
    print(f"r3 = {r3}")
except Exception as e:
    print(f"Error in computing r3: {e}")
    e3_arr = []
    r3 = None

from EV_CS_Model_Module import EV_CS_Solve
# تولید جبهه پارتو با ε-Constraint
pareto_solution = []
var_collection=[]

for e2 in e2_arr:
    for e3 in e3_arr:
        try:
            output = EV_CS_Solve(index_obj=0, e2_val=e2, e3_val=e3, r2=r2, r3=r3,epsilon=0.01,gap=0.00)
            result=output[0]
            var=output[1]

            if result:
                pareto_solution.append((result['f1'], result['f2'], result['f3']))
                var_collection.append(var)
                
                print(f"✓ f1={result['f1']:.2f} for f2={result['f2']:.2f}, f3={result['f3']:.2f}")


        except Exception as e:
            print(f"✗ Error for f2={result['f2']:.2f}, f3={result['f3']:.2f}: {e}")

# نمایش خروجی نهایی
print("\nPareto Solutions:")
for sol in pareto_solution:
    print(f"f1: {sol[0]:.2f}, f2: {sol[1]:.2f}, f3: {sol[2]:.2f}")

final_result=mcdm_rank_analysis(pareto_solution, weights=(0.5, 0.25, 0.25), vikor_v=0.5, electre_threshold=0.65)
# print solutions
final_result.to_csv("mcdm_results.csv", index=False)
# print related variables
order = (final_result['Option'].to_numpy() - 1).tolist()
n = len(var_collection)
sorted_var_collection = [var_collection[i] for i in order if 0 <= i < n]

with pd.ExcelWriter("output.xlsx") as writer:
    for idx, dic in enumerate(sorted_var_collection):
        # ساخت دیتافریم از دیکشنری
        df = pd.DataFrame(dic.items(), columns=["Variable", "Value"])
        
        # حذف مقادیر 0 یا None
        df = df[(df["Value"] != 0) & (df["Value"].notna())]
        
        # نوشتن در شیت با نام اندیس
        df.to_excel(writer, sheet_name=f"{idx+1}", index=False)


import matplotlib.pyplot as plt
import seaborn as sns

# --- Pareto Front: Profit vs CO₂ ---
pareto_df = pd.DataFrame(pareto_solution, columns=["Profit","CO2","GridDep"])
plt.figure(figsize=(7,5))
sc = plt.scatter(pareto_df["CO2"], pareto_df["Profit"], 
                 c=pareto_df["GridDep"], cmap="viridis", s=80)
plt.colorbar(sc, label="Grid Dependency (%)")
for i, (x,y) in enumerate(zip(pareto_df["CO2"], pareto_df["Profit"])):
    plt.text(x, y, str(i+1), fontsize=8)
plt.xlabel("CO₂ Emissions (tons)")
plt.ylabel("Profit (k€)")
plt.title("Pareto Front: Profit vs CO₂ (36 solutions)")
plt.tight_layout()
plt.show()

rank_cols = [
    "SAW_Rank",
    "TOPSIS_Rank",
    "PROMETHEE_II_Rank",
    "ELECTRE_I_Rank",
    "MOORA_Rank",
    "VIKOR_Rank"
]
plt.figure(figsize=(10,8))
sns.heatmap(
    final_result.set_index("Option")[rank_cols],
    cmap="YlOrRd",    # yellow-red gradient
    cbar_kws={'label': 'Rank'},
    annot=False
)

# Overlay dots for rank positions
for i, option in enumerate(final_result["Option"]):
    for j, col in enumerate(rank_cols):
        rank_val = final_result.loc[final_result["Option"]==option, col].values[0]
        plt.text(j+0.5, i+0.5, "●", ha="center", va="center", color="black", fontsize=10)

plt.title("MCDM Ranking Comparison Across Methods (Dot Overlay)", fontsize=14, pad=15)
plt.ylabel("Solution ID", fontsize=12)
plt.xlabel("MCDM Method", fontsize=12)
plt.tight_layout()
plt.show()

# --- (Optional) 3D Pareto Scatter ---
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pareto_df["CO2"], pareto_df["GridDep"], pareto_df["Profit"], c='blue', s=60)
ax.set_xlabel("CO₂ Emissions")
ax.set_ylabel("Grid Dependency (%)")
ax.set_zlabel("Profit (k€)")
ax.set_title("3D Trade-off Visualization (36 solutions)")
plt.show()
   
# --- Tornado-style sensitivity: SAW and Final_Rank (score) for ONE solution ---
import numpy as np
import matplotlib.pyplot as plt
from MCDM import mcdm_rank_analysis

def renorm(w):
    """Clip negatives, renormalize to sum=1; fallback to equal if sum=0."""
    w = np.maximum(np.array(w, dtype=float), 0.0)
    s = w.sum()
    return (w / s) if s > 0 else np.array([1/3, 1/3, 1/3])

# Baseline weights (Profit, CO2, Grid) — adjust if you like
baseline = np.array([0.5, 0.25, 0.25])
delta    = 0.15  # weight tweak size for ± scenarios

# 1) Pick the solution to analyze:
#    auto-pick: best Final_Rank under baseline
res_base = mcdm_rank_analysis(pareto_solution, weights=tuple(baseline),
                              vikor_v=0.5, electre_threshold=0.65)
best_id  = int(res_base.sort_values("Final_Rank").iloc[0]["Option"])
solution_id = best_id     # or set manually, e.g. solution_id = 5

# 2) Baseline metrics for the chosen solution
saw0   = float(res_base.loc[res_base["Option"] == solution_id, "SAW"])
rank0  = float(res_base.loc[res_base["Option"] == solution_id, "Final_Rank"])
Nopts  = len(pareto_solution)
score0 = Nopts + 1 - rank0  # convert rank to "bigger is better" score

labels, d_saw, d_score = [], [], []
names = ["Profit weight", "CO₂ weight", "Grid weight"]

# 3) +/- delta on each weight (renormalized)
for i, name in enumerate(names):
    up = renorm(baseline + np.eye(3)[i] * delta)
    dn = renorm(np.clip(baseline - np.eye(3)[i] * delta, 0, None))

    res_up = mcdm_rank_analysis(pareto_solution, weights=tuple(up),
                                vikor_v=0.5, electre_threshold=0.65)
    res_dn = mcdm_rank_analysis(pareto_solution, weights=tuple(dn),
                                vikor_v=0.5, electre_threshold=0.65)

    saw_up = float(res_up.loc[res_up["Option"] == solution_id, "SAW"])
    saw_dn = float(res_dn.loc[res_dn["Option"] == solution_id, "SAW"])

    r_up = float(res_up.loc[res_up["Option"] == solution_id, "Final_Rank"])
    r_dn = float(res_dn.loc[res_dn["Option"] == solution_id, "Final_Rank"])
    score_up = Nopts + 1 - r_up
    score_dn = Nopts + 1 - r_dn

    labels += [f"{name} (+Δ)", f"{name} (−Δ)"]
    d_saw  += [saw_up - saw0,  saw_dn - saw0]
    d_score+= [score_up - score0, score_dn - score0]
# --- Utility: normalize weights ---
def renorm(w):
    """Clip negatives, renormalize to sum=1; fallback to equal if sum=0."""
    w = np.maximum(np.array(w, dtype=float), 0.0)
    s = w.sum()
    return (w / s) if s > 0 else np.array([1/3, 1/3, 1/3])

# --- Parameters ---
baseline = np.array([0.5, 0.25, 0.25])   # Profit, CO2, Grid
delta    = 0.3                         # weight perturbation size
methods  = ["SAW", "TOPSIS", "VIKOR", "ELECTRE_I", "MOORA", "PROMETHEE_II"]
names    = ["Profit weight", "CO₂ weight", "Grid weight"]

# --- 1) Baseline analysis ---
res_base = mcdm_rank_analysis(pareto_solution, weights=tuple(baseline),
                              vikor_v=0.5, electre_threshold=0.65)

# pick best solution according to Final_Rank
best_id = int(res_base.sort_values("Final_Rank").iloc[0]["Option"])
solution_id = best_id
Nopts = len(pareto_solution)

# store baseline scores for each method
baseline_vals = {m: float(res_base.loc[res_base["Option"] == solution_id, m]) 
                 for m in methods}
baseline_rank = float(res_base.loc[res_base["Option"] == solution_id, "Final_Rank"])
baseline_score = Nopts + 1 - baseline_rank   # bigger = better

# --- 2) Sensitivity containers ---
labels = []
delta_vals = {m: [] for m in methods}
delta_score = []

# --- 3) Perturb each weight ---
for i, name in enumerate(names):
    up = renorm(baseline + np.eye(3)[i] * delta)
    dn = renorm(np.clip(baseline - np.eye(3)[i] * delta, 0, None))

    # recalc with perturbed weights
    res_up = mcdm_rank_analysis(pareto_solution, weights=tuple(up),
                                vikor_v=0.5, electre_threshold=0.65)
    res_dn = mcdm_rank_analysis(pareto_solution, weights=tuple(dn),
                                vikor_v=0.5, electre_threshold=0.65)

    # scores for each method
    for m in methods:
        up_val = float(res_up.loc[res_up["Option"] == solution_id, m])
        dn_val = float(res_dn.loc[res_dn["Option"] == solution_id, m])
        delta_vals[m] += [up_val - baseline_vals[m],
                          dn_val - baseline_vals[m]]

    # rank sensitivity (using Final_Rank)
    r_up = float(res_up.loc[res_up["Option"] == solution_id, "Final_Rank"])
    r_dn = float(res_dn.loc[res_dn["Option"] == solution_id, "Final_Rank"])
    score_up = Nopts + 1 - r_up
    score_dn = Nopts + 1 - r_dn
    delta_score += [score_up - baseline_score, score_dn - baseline_score]

    labels += [f"{name} (+Δ)", f"{name} (−Δ)"]

# --- 4) Plot sensitivity tornado plots ---
fig, axes = plt.subplots(3, 2, figsize=(15, 8), sharex=False)

for ax, m in zip(axes.flatten(), methods):
    order = np.argsort(np.abs(delta_vals[m]))[::-1]  # sort by biggest effect
    ax.barh([labels[j] for j in order], [delta_vals[m][j] for j in order])
    ax.axvline(0, color="k", linewidth=1)
    ax.set_title(f"{m} Sensitivity (Solution {solution_id})")
    ax.set_xlabel(f"Δ {m} vs baseline")

plt.suptitle(f"Sensitivity Analysis across 6 MCDM Methods\nBaseline={tuple(baseline)}",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# 5) Plot tornado for Final_Rank-derived score (bigger = better)
order2 = np.argsort(np.abs(d_score))[::-1]
plt.figure(figsize=(8,5))
plt.barh([labels[j] for j in order2], [d_score[j] for j in order2])
plt.axvline(0, color="k", linewidth=1)
plt.title(f"Final_Rank Sensitivity (as Score)  Solution {solution_id}")
plt.xlabel("Δ Score (N+1−Final_Rank) vs baseline")
plt.tight_layout(); plt.show()


# --- Sensitivity Analysis of Weight Scenarios (bar chart of best alternative) ---
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import OrderedDict
from MCDM import mcdm_rank_analysis

# 1) Define your weight scenarios (Profit, CO2, Grid)
methods = ["SAW", "TOPSIS", "VIKOR", "ELECTRE_I", "MOORA", "PROMETHEE_II"]
# Define scenarios
scenarios = OrderedDict([
    ("(0.33, 0.33, 0.33)", (0.33, 0.33, 0.33)),
    ("(0.5, 0.25, 0.25)",  (0.5,  0.25, 0.25)),
    ("(0.25, 0.5, 0.25)",  (0.25, 0.5,  0.25)),
    ("(0.25, 0.25, 0.5)",  (0.25, 0.25, 0.5)),
    ("(0.6, 0.2, 0.2)",    (0.6,  0.2,  0.2)),
    ("(0.2, 0.4, 0.4)",    (0.2,  0.4,  0.4)),
])
rows = []
for label, w in scenarios.items():
    res = mcdm_rank_analysis(
        pareto_solution,
        weights=w,
        vikor_v=0.5,
        electre_threshold=0.65
    )
    # Best alternative by Final_Rank
    best = res.sort_values("Final_Rank").iloc[0]

    row = {
        "Scenario": label,
        "Weights": w,
        "Best_Option": int(best["Option"]),
        "Best_Final_Rank": float(best["Final_Rank"]),
        "Best_Final_Score": float(best["Final_Score"])
    }
    for m in methods:
        row[f"Best_{m}"] = float(best[m])
    rows.append(row)

sens_df = pd.DataFrame(rows)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)

for ax, m in zip(axes.flatten(), methods):
    y_labels =  [f"Scenario {i+1}" for i in range(len(sens_df))]
    x_vals   = sens_df[f"Best_{m}"].tolist()

    bars = ax.barh(y_labels, x_vals, alpha=0.9)
    ax.set_xlabel(f"{m} Score of Best Alternative")
    ax.set_title(f"{m} Sensitivity")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Annotate “Best: Option X” at the end of each bar
    for i, (x, opt) in enumerate(zip(x_vals, sens_df["Best_Option"])):
        ax.text(x + 0.01, i, f"Best: {opt}", va="center")

plt.suptitle("Sensitivity Analysis of Weight Scenarios across MCDM Methods", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# 4) (Optional) Save a tidy CSV summary you can cite in your paper
sens_df.to_csv("sensitivity_weight_scenarios_summary.csv", index=False)
print("Saved: sensitivity_weight_scenarios_summary.csv")
print(sens_df)

#----------------------------------------------------------------------------------

# Suppose final_result is your MCDM ranking table
# The best solution is the first row (ranked highest)
import re

# Reload df_best just in case
df_best = pd.read_excel("best_solution.xlsx", sheet_name="BestSolution")

# Extract technology prefix (everything before the first "[")
df_best["Tech"] = df_best["Variable"].str.extract(r"^([a-zA-Z0-9_]+)")

# List of flows we care about
flow_vars = ["pg2v", "pv2g", "p_chp", "h_bo", "h_h2p", "h_p2h"]
colors = {
    "pg2v": "#f4b400",   # orange-yellow
    "pv2g": "#66b3ff",   # light blue
    "h_h2p": "#2ca02c",    # green
    "h_p2h": "#ffeb99",    # pale yellow-green
    "p_chp": "#ff7043",  # coral
    "h_bo": "#8c564b",   # brown
}
# Aggregate total over all i, j, t
df_share = (
    df_best[df_best["Tech"].isin(flow_vars)]
    .groupby("Tech")["Value"]
    .sum()
)

# Take absolute values (to avoid negatives cancelling out)
df_share = df_share.abs()

# --- Pie chart ---
plt.figure(figsize=(8,8))
plt.pie(
    df_share,
    labels=df_share.index,
    autopct="%.1f%%",
    startangle=90,
    colors=plt.cm.Set3.colors
)
plt.title("Energy Contribution Share by Technology (Best Solution)")
plt.show()

# --- Print numerical totals too ---
print("🔹 Total contributions (summed over all i, j, t):")
print(df_share)

#------------------------------------------------------------------------------------
import re

# --- Parse variables into tech, i, j, t
def parse_var(name):
    if "[" in name:
        tech, inside = name.split("[", 1)
        tech = tech.lower()
        nums = [int(x) for x in inside.strip("]").split(",")]
        if len(nums) == 3:
            i, j, t = nums
        elif len(nums) == 2:
            j, t = nums
            i = None
        else:
            i = j = t = None
    else:
        tech, i, j, t = name.lower(), None, None, None
    return tech, i, j, t

parsed = df_best["Variable"].apply(parse_var)
df_best[["Tech","i","j","time"]] = pd.DataFrame(parsed.tolist(), index=df_best.index)
df_best["Value"] = pd.to_numeric(df_best["Value"], errors="coerce")

# --- Force lowercase for safety
df_best["Tech"] = df_best["Tech"].str.lower()

# Redefine vars
elec_vars = ["pg2v", "pv2g"]
therm_vars = ["h_h2p", "h_p2h", "p_chp", "h_bo"]

# Aggregate by time (now works since we parsed 'time')
df_elec = (
    df_best[df_best["Tech"].isin(elec_vars)]
    .pivot_table(index="time", columns="Tech", values="Value", aggfunc="sum")
    .fillna(0)
)

df_therm = (
    df_best[df_best["Tech"].isin(therm_vars)]
    .pivot_table(index="time", columns="Tech", values="Value", aggfunc="sum")
    .fillna(0)
)

# Colors
colors = {
    "pg2v": "#f4b400",   # orange-yellow
    "pv2g": "#66b3ff",   # light blue
    "h_h2p": "#2ca02c",  # green
    "h_p2h": "#ffeb99",  # pale yellow-green
    "p_chp": "#ff7043",  # coral
    "h_bo": "#8c564b",   # brown
}


# --- Plot twin axes
fig, ax1 = plt.subplots(figsize=(14,7))

df_elec.plot(
    kind="bar", stacked=True, ax=ax1,
    color=[colors[c] for c in df_elec.columns], alpha=0.85
)
ax1.set_ylabel("Electricity Flow (kW)", fontsize=12)
ax1.set_xlabel("Time (hour)", fontsize=12)

ax2 = ax1.twinx()
df_therm.plot(
    kind="bar", stacked=True, ax=ax2,
    color=[colors[c] for c in df_therm.columns], alpha=0.6
)
ax2.set_ylabel("Thermal Flow (kW)", fontsize=12)

# Legends
handles1 = [plt.Rectangle((0,0),1,1, color=colors[c]) for c in df_elec.columns]
handles2 = [plt.Rectangle((0,0),1,1, color=colors[c]) for c in df_therm.columns]
ax1.legend(handles1, df_elec.columns, title="Electricity", loc="upper left")
ax2.legend(handles2, df_therm.columns, title="Thermal", loc="upper right")

ax1.set_title("Electricity & Thermal Flows (Best Solution)", fontsize=14)
plt.tight_layout()
plt.show()
#--------------------------------------------------------------------------------------------
# Variables for ramp up/down
ramp_vars = ["p_rampup", "p_rampdn", "h_rampup", "h_rampdn"]

# Pivot the data
df_ramp = (
    df_best[df_best["Tech"].isin(ramp_vars)]
    .pivot_table(
        index="time", 
        columns="Tech",   # now using parsed tech name
        values="Value", 
        aggfunc="sum"
    )
    .fillna(0)
    .sort_index()
)

# Define custom colors (same as shown in figure)
colors = {
    "p_rampup": "#f4b400",  # orange-yellow
    "p_rampdn": "#66b3ff",  # light blue
    "h_rampup": "#2ca02c",  # green
    "h_rampdn": "#ffeb99",  # pale yellow-green
}

# Plot stacked bar chart
ax = df_ramp.plot(
    kind="bar",
    stacked=True,
    figsize=(14,7),
    color=[colors[c] for c in df_ramp.columns],
    alpha=0.9
)

ax.set_xlabel("Time (hour)")
ax.set_ylabel("Power change (kW)")
ax.set_title("Ramp-Up and Ramp-Down (Bar Chart, Best Solution)")
plt.legend(title="Variable", loc="upper left")

plt.tight_layout()
plt.show()
#---------------------------------------------------------------------------------------
# --- Sample data (replace with your real optimized capacities) ---
data = {
    "Station": ["Station 1 (Nano)", "Station 2 (Nano)", "Station 3 (Micro)", "Station 4 (Micro)", "Station 5 (Micro)"],
    "CHP":    [150, 180, 600, 750, 800],
    "Boiler": [100, 110, 500, 600, 650],
    "ESS":    [200, 220, 800, 900, 1000],
    "HSS":    [80, 90, 300, 350, 400],
    "TS":     [100, 120, 400, 450, 500],
    "PV":     [250, 260, 900, 950, 1000],
}
df = pd.DataFrame(data)

# --- Plot grouped bar chart ---
facilities = ["CHP", "Boiler", "ESS", "HSS", "TS", "PV"]
x = np.arange(len(df["Station"]))  # station indices
width = 0.12  # bar width

fig, ax = plt.subplots(figsize=(16,8))

for i, facility in enumerate(facilities):
    ax.bar(x + i*width, df[facility], width, label=facility)

# Formatting
ax.set_xticks(x + width*(len(facilities)/2))
ax.set_xticklabels(df["Station"], rotation=30, ha="right")
ax.set_ylabel("Installed Capacity (kW / kWh)")
ax.set_title("Installed Capacities of Facilities Across 5 Stations (Sample Data)")
ax.legend(title="Facility")

plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------
df_obj = pd.DataFrame(pareto_solution, columns=["f1", "f2", "f3"])

# Compute best (max for f1=profit, min for f2,f3=costs)
best_values = {
    "Profit (f1)": df_obj["f1"].max(),
    "CO₂ Emission (f2)": df_obj["f2"].min(),
    "Grid Dependency (f3)": df_obj["f3"].min()
}

# Compute worst (min for f1, max for f2,f3)
worst_values = {
    "Profit (f1)": df_obj["f1"].min(),
    "CO₂ Emission (f2)": df_obj["f2"].max(),
    "Grid Dependency (f3)": df_obj["f3"].max()
}

# Normalized range = Best - Worst
normalized_range = {
    k: best_values[k] - worst_values[k] for k in best_values
}

# Build payoff matrix dataframe
payoff_matrix = pd.DataFrame({
    "Objective": ["Profit (f1)", "CO₂ Emission (f2)", "Grid Dependency (f3)"],
    "Best Value": [best_values["Profit (f1)"], best_values["CO₂ Emission (f2)"], best_values["Grid Dependency (f3)"]],
    "Worst Value": [worst_values["Profit (f1)"], worst_values["CO₂ Emission (f2)"], worst_values["Grid Dependency (f3)"]],
    "Normalized Range": [normalized_range["Profit (f1)"], normalized_range["CO₂ Emission (f2)"], normalized_range["Grid Dependency (f3)"]],
})

print(payoff_matrix)

# Optional: save to Excel
payoff_matrix.to_excel("payoff_matrix.xlsx", index=False)

# Force lowercase for consistency
df_best["Tech"] = df_best["Tech"].str.lower()

# Redefine vars
elec_vars = ["pg2v", "pv2g"]
therm_vars = ["h2p", "p2h", "p_chp", "h_bo"]

# Aggregate by time
df_elec = (
    df_best[df_best["Tech"].isin(elec_vars)]
    .pivot_table(index="time", columns="Tech", values="Value", aggfunc="sum")
    .fillna(0)
)

df_therm = (
    df_best[df_best["Tech"].isin(therm_vars)]
    .pivot_table(index="time", columns="Tech", values="Value", aggfunc="sum")
    .fillna(0)
)

# Consistent colors
colors = {
    "pg2v": "#f4b400",   # orange-yellow
    "pv2g": "#66b3ff",   # light blue
    "h2p": "#2ca02c",    # green
    "p2h": "#ffeb99",    # pale yellow-green
    "p_chp": "#ff7043",  # coral
    "h_bo": "#8c564b",   # brown
}

fig, ax1 = plt.subplots(figsize=(14,7))

# --- Electricity (left axis)
df_elec.plot(
    kind="bar",
    stacked=True,
    ax=ax1,
    color=[colors[c] for c in df_elec.columns],
    alpha=0.85
)
ax1.set_ylabel("Electricity Flow (kW)", fontsize=12)
ax1.set_xlabel("Time (hour)", fontsize=12)

# --- Thermal (right axis)
ax2 = ax1.twinx()
df_therm.plot(
    kind="bar",
    stacked=True,
    ax=ax2,
    color=[colors[c] for c in df_therm.columns],
    alpha=0.6
)
ax2.set_ylabel("Thermal Flow (kW)", fontsize=12)

# --- Legends with correct colors
handles1 = [plt.Rectangle((0,0),1,1, color=colors[c]) for c in df_elec.columns]
handles2 = [plt.Rectangle((0,0),1,1, color=colors[c]) for c in df_therm.columns]
ax1.legend(handles1, df_elec.columns, title="Electricity", loc="upper left")
ax2.legend(handles2, df_therm.columns, title="Thermal", loc="upper right")

# --- Title
ax1.set_title("Electricity & Thermal Flows (Best Solution)", fontsize=14)

plt.tight_layout()
plt.show()  