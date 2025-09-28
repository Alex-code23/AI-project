import yfinance as yf
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os

# --- Replace your symbol_dict here or import it ---
symbol_dict = {
    "TTE": "Total",
    "XOM": "Exxon",
    "CVX": "Chevron",
    "COP": "ConocoPhillips",
    "VLO": "Valero Energy",
    "MSFT": "Microsoft",
    "IBM": "IBM",
    "CMCSA": "Comcast",
    "DELL": "Dell",
    "HPQ": "HP",
    "AMZN": "Amazon",
    "TM": "Toyota",
    "F": "Ford",
    "HMC": "Honda",
    "NOC": "Northrop Grumman",
    "BA": "Boeing",
    "KO": "Coca Cola",
    "MMM": "3M",
    "MCD": "McDonald's",
    "PEP": "Pepsi",
    "K": "Kellogg",
    "MAR": "Marriott",
    "PG": "Procter Gamble",
    "CL": "Colgate-Palmolive",
    "GE": "General Electrics",
    "WFC": "Wells Fargo",
    "JPM": "JPMorgan Chase",
    "AIG": "AIG",
    "AXP": "American express",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "AAPL": "Apple",
    "SAP": "SAP",
    "CSCO": "Cisco",
    "TXN": "Texas Instruments",
    "XRX": "Xerox",
    "WMT": "Wal-Mart",
    "HD": "Home Depot",
    "GSK": "GlaxoSmithKline",
    "PFE": "Pfizer",
    "SNY": "Sanofi-Aventis",
    "NVS": "Novartis",
    "KMB": "Kimberly-Clark",
    "R": "Ryder",
    "GD": "General Dynamics",
    "CVS": "CVS",
    "CAT": "Caterpillar",
    "DD": "DuPont de Nemours",
    # Ajouts 
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "META": "Meta Platforms",
    "GOOGL": "Alphabet (Class A)",
    "NFLX": "Netflix",
    "ADBE": "Adobe",
    "INTU": "Intuit",
    "PYPL": "PayPal",
    "UBER": "Uber",
    "LYFT": "Lyft",
    "SHOP": "Shopify",
    "SPOT": "Spotify",
    "BRK-B": "Berkshire Hathaway (Class B)",
    "BP": "BP plc",
    "AMGN": "Amgen",
    "BABA": "Alibaba",
    "TSM": "Taiwan Semiconductor Manufacturing",
    "ORCL": "Oracle"

}

# Ordered arrays as in your original code
symbols, names = np.array(sorted(symbol_dict.items())).T

# User-configurable options
start = "2021-01-01"   # or set None and use period below
end = "2023-12-31"     # e.g. "2023-12-31" or None
# 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
period = "3mo"         # e.g. "5y" - if not None, start/end are ignored
align = "intersection" # "intersection" (common dates) or "union" (outer join, may contain NaNs)
save_csv = True        # save each ticker history as CSV
outdir = "stats/stock_market/quotes_yf"   # directory to save CSVs

os.makedirs(outdir, exist_ok=True)

dfs = []   # list of (symbol, df) for successfully downloaded tickers
failed = []

# Download loop (using tqdm for progress)
for symbol in tqdm(symbols, desc="Downloading"):
    try:
        # yfinance download: either interval + period, or start/end
        if period is not None:
            df = yf.download(symbol, period=period, progress=False, auto_adjust=False)
        else:
            df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)

        # df can be empty if ticker not found or no data
        if df is None or df.shape[0] == 0:
            print(f"Warning: no data for {symbol}", file=sys.stderr)
            failed.append(symbol)
            continue

        # Normalize column names to lowercase to match your code
        df.columns = [c[0].lower() for c in df.columns]

        # Keep at least 'open' and 'close' columns if available
        if "open" not in df.columns or "close" not in df.columns:
            print(f"Warning: missing open/close for {symbol}, columns: {df.columns}", file=sys.stderr)
            failed.append(symbol)
            continue

        # Optionally save CSV
        if save_csv:
            csv_path = os.path.join(outdir, f"{symbol}.csv")
            # Save index (dates) as a column
            df.to_csv(csv_path)

        # Keep the df (with date index)
        dfs.append((symbol, df[["open", "close"]].copy()))

    except Exception as e:
        print(f"Error downloading {symbol}: {e}", file=sys.stderr)
        failed.append(symbol)

print(f"Downloaded {len(dfs)} tickers, failed {len(failed)} tickers.", file=sys.stderr)
if failed:
    print("Failed tickers:", failed, file=sys.stderr)

if len(dfs) == 0:
    raise SystemExit("No data downloaded. Aborting.")

# Align time indices
if align == "intersection":
    common_index = dfs[0][1].index
    # for symbol, df in dfs[1:]:
    #     common_index = common_index.intersection(df.index)

    for sym, df in dfs[1:]:
        old = len(common_index)
        common_index = common_index.intersection(df.index)
        if len(common_index) != old:
            print(f"Intersection reduced by {old-len(common_index)} when adding {sym}")

    common_index = common_index.sort_values()
    print(f"Using intersection of dates -> {len(common_index)} dates", file=sys.stderr)

    # Build matrices in the same date order
    close_prices = np.vstack([df.loc[common_index, "close"].values for _, df in dfs])
    open_prices = np.vstack([df.loc[common_index, "open"].values for _, df in dfs])
    dates = common_index

elif align == "union":
    # Outer join all dfs on the date index
    all_df = None
    for symbol, df in dfs:
        tmp = df.rename(columns={"open": f"open_{symbol}", "close": f"close_{symbol}"})
        if all_df is None:
            all_df = tmp
        else:
            all_df = all_df.join(tmp, how="outer")
    # Option: forward-fill/backfill or drop rows with NaN
    all_df = all_df.sort_index().ffill().bfill()  # fill missing values
    # Extract back to matrices
    close_cols = [c for c in all_df.columns if c.startswith("close_")]
    open_cols = [c for c in all_df.columns if c.startswith("open_")]
    close_prices = all_df[close_cols].values.T
    open_prices = all_df[open_cols].values.T
    dates = all_df.index
    print(f"Using union of dates -> {len(dates)} dates (filled NaNs).", file=sys.stderr)

else:
    raise ValueError("align must be 'intersection' or 'union'")

print(dfs[0][1].head())

# The daily variations of the quotes are what carry the most information
variation = close_prices - open_prices
print(f"Variation matrix shape: {variation.shape} (stocks, days)", file=sys.stderr)
print(f"Dates from {dates[0]} to {dates[-1]}", file=sys.stderr)

from sklearn import covariance

# valeurs candidates du paramètre de régularisation que GraphicalLassoCV va tester
# plus alphas est grand, plus la matrice de précision estimée sera "creuse" (plus d'éléments nuls) donc le graphe aura moins d'arêtes
alphas = np.logspace(-1, 2, num=20)
# Objectif : estimer la matrice de précision Θ=Σ**−1  (inverse de la covariance).
edge_model = covariance.GraphicalLassoCV(alphas=alphas)

# standardize the time series: using correlations rather than covariance
# former is more efficient for structure recovery
X = variation.copy().T  # shape (n_samples, n_features) = (days, stocks)
X /= X.std(axis=0) # standardize features (stocks)
edge_model.fit(X)

from sklearn import cluster

# Cluster using affinity propagation
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
n_labels = labels.max()

for i in range(n_labels + 1):
    print(f"Cluster {i + 1}: {', '.join(names[labels == i])}")

# Finding a low-dimension embedding for visualization: find the best position of
# the nodes (the stocks) on a 2D plane

from sklearn import manifold

# node_position_model = manifold.LocallyLinearEmbedding(n_components=2, eigen_solver="dense", n_neighbors=10)
# another option: manifold.MDS
# node_position_model = manifold.MDS(n_components=2, random_state=0)
# node_position_model = manifold.SpectralEmbedding(n_components=2, random_state=0)
# t-SNE
node_position_model = manifold.TSNE(n_components=2, random_state=0,init="random", perplexity=30,max_iter=400)

embedding = node_position_model.fit_transform(X.T).T

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.figure(1, facecolor="w", figsize=(10, 8))
plt.clf()
ax = plt.axes([0.0, 0.0, 1.0, 1.0])
plt.axis("off")

# Plot the graph of partial correlations
partial_correlations = edge_model.precision_.copy()
# normalize the precision matrix to a correlation matrix
# partial correlation between i and j is -Θ_ij / sqrt(Θ_ii * Θ_jj)
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = np.abs(np.triu(partial_correlations, k=1)) > 0.05

# Plot the nodes using the coordinates of our embedding
plt.scatter(
    embedding[0], embedding[1], s=70 * d**2, c=labels, cmap=plt.cm.nipy_spectral
)

# Plot the edges
start_idx, end_idx = non_zero.nonzero()
# a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [
    [embedding[:, start], embedding[:, stop]] for start, stop in zip(start_idx, end_idx)
]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(
    segments, zorder=0, cmap=plt.cm.hot_r, norm=plt.Normalize(0, 0.7 * values.max())
)
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(zip(names, labels, embedding.T)):
    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = "left"
        x = x + 0.002
    else:
        horizontalalignment = "right"
        x = x - 0.002
    if this_dy > 0:
        verticalalignment = "bottom"
        y = y + 0.002
    else:
        verticalalignment = "top"
        y = y - 0.002
    plt.text(
        x,
        y,
        name,
        size=5,
        horizontalalignment=horizontalalignment,
        verticalalignment=verticalalignment,
        bbox=dict(
            facecolor="w",
            edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
            alpha=0.6,
        ),
    )

plt.xlim(
    embedding[0].min() - 0.15 * np.ptp(embedding[0]),
    embedding[0].max() + 0.10 * np.ptp(embedding[0]),
)
plt.ylim(
    embedding[1].min() - 0.03 * np.ptp(embedding[1]),
    embedding[1].max() + 0.03 * np.ptp(embedding[1]),
)

plt.show()