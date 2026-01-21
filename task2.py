import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_main = r"C:\Users\ADMIN\Desktop\internship task\task 2\Unemployment in India.csv"
file_state = r"C:\Users\ADMIN\Desktop\internship task\task 2\Unemployment_Rate_upto_11_2020.csv"

df_main = pd.read_csv(file_main)
df_state = pd.read_csv(file_state)

print("Main data shape:", df_main.shape)
print("State data shape:", df_state.shape)

df_main.columns = [c.strip().replace(" ", "_").lower() for c in df_main.columns]
df_state.columns = [c.strip().replace(" ", "_").lower() for c in df_state.columns]

print("\nMain columns:", df_main.columns)
print("State columns:", df_state.columns)

df_main.rename(
    columns={"estimated_unemployment_rate_(%)": "unemp_rate"},
    inplace=True
)
df_state.rename(
    columns={"estimated_unemployment_rate_(%)": "unemp_rate"},
    inplace=True
)

df_main["date"] = pd.to_datetime(df_main["date"], dayfirst=True, errors="coerce")
df_state["date"] = pd.to_datetime(df_state["date"], dayfirst=True, errors="coerce")

print("\nMain head:")
print(df_main.head())
print("\nState head:")
print(df_state.head())

national_trend = (
    df_main.groupby("date")["unemp_rate"]
    .mean()
    .reset_index()
    .sort_values("date")
)

plt.figure(figsize=(10, 5))
sns.lineplot(data=national_trend, x="date", y="unemp_rate", marker="o")
plt.title("Average Unemployment Rate Over Time (National)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

if "area" in df_main.columns:
    rural_urban_trend = (
        df_main.groupby(["date", "area"])["unemp_rate"]
        .mean()
        .reset_index()
        .sort_values("date")
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=rural_urban_trend,
        x="date",
        y="unemp_rate",
        hue="area",
        marker="o"
    )
    plt.title("Rural vs Urban Unemployment Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

latest_date = df_state["date"].max()
df_latest = df_state[df_state["date"] == latest_date].copy()

df_latest = df_latest.sort_values("unemp_rate", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(
    data=df_latest,
    x="unemp_rate",
    y="region",
    palette="viridis"
)
plt.title(f"State-wise Unemployment Rate on {latest_date.date()}")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("State")
plt.tight_layout()
plt.show()

state_avg = (
    df_state.groupby("region")["unemp_rate"]
    .mean()
    .reset_index()
    .sort_values("unemp_rate", ascending=False)
)

print("\nTop 5 states by average unemployment rate in dataset:")
print(state_avg.head(5))

plt.figure(figsize=(8, 5))
sns.barplot(
    data=state_avg.head(5),
    x="unemp_rate",
    y="region",
    palette="magma"
)
plt.title("Top 5 States by Average Unemployment Rate")
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("State")
plt.tight_layout()
plt.show()
