# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Step 1: beam A (ON+OFF) の校正・LPF・FA+ICA
#
# `tau0/empirical_tau0.py` のStep 0-3.5 (すでにdaisy/pswsc両対応で書かれて
# いる) をベースに、以下を変更する:
# - 対象を小さいNGC1068のpswsc観測 (obsid=20240716104712, beam A+B, ~137秒)
# - beam Aの **ON+OFF両方** を使う (元のOFF_NOD_ONLYはPWVフィット用にOFFだけ
#   を残す設計だったが、今回はON/OFFチョップに相関するICを探すのが目的なので
#   両方必要)
# - 後段 (chop-IC同定・ON/OFF較正) のために、各サンプルのstate ("ON"/"OFF")
#   ラベルを別途保持しておく

# %%
import gc
import sys

sys.path.insert(0, "../src")  # editable installなしでも pswsc_faica_pipeline を import できるように

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FactorAnalysis
from tqdm import tqdm

import decode as dc
from d24_tools import reduce, utils
from d24_tools.atm import get_pwv, get_eta_atm
from d24_tools.fitting import pwv_fit_func

import matplotlib.pyplot as plt

from pswsc_faica_pipeline.faica import whiten_fa, jade

# %%
DEMS_PATH = "/home/shiro/pyenv/deshima/data/v20250826/dems_rm_neg/dems_20240715094140.zarr.zip"
SCAN_STATES = ["ON", "OFF"]

# 校正を借用する専用skydip (対象pswscの131.98分前、QC通過済み、R2_median=0.996)。
# この観測自身のmini-skydip/OFF点校正はどちらもelevation変化が弱すぎて破綻した
# (詳細はREADME/memory参照)。empirical eta_atm (skydip由来) とmodelの比較で、
# elevationの両端 (35度/85度) でempirical > modelの系統的なずれ (両端で同じ符号、
# MAEが中央部の5-6倍) が見つかっており、線形校正が非線形な真の関係を近似した
# ことによる残差の可能性が高いと判断し、2次校正を採用する。
REFERENCE_SKYDIP_PATH = "/home/shiro/pyenv/deshima/data/v20250826/dems_rm_neg/dems_20240715072703.zarr.zip"

N_FACTORS = 5
N_JADE_SAMPLES = None  # Noneなら全サンプル使用

LPF_CUTOFF_HZ = 0.1
LPF_ORDER = 4

R2_THRES = 0.85
CHAN_PLOT = 100  # 可視化用チャンネル (0-based index)

# %%
# --- Step 0: DEMS読み込み (df/f 必須) ---------------------------------------

da = dc.load.dems(DEMS_PATH)
if da.long_name != "df/f":
    da = dc.load.dems(DEMS_PATH, data_scaling="df/f")
print(f"data scaling: {da.long_name}")
obsid = str(da.aste_obs_id)
print(f"obsid={obsid}, object={da.attrs.get('object')}, N_time={da.time.size}")

# %%
# --- Step 0.5: 不良チャンネル除去 -------------------------------------------

arr = da.to_numpy()
bad = np.isnan(np.mean(arr, axis=0)) | (np.nanstd(arr, axis=0) == 0)
print(f"removing {bad.sum()} / {bad.size} bad channels (NaN or constant)")
da = da[:, ~bad]
del arr
gc.collect()

# %%
# --- Step 1: despike --------------------------------------------------------
# 校正は近傍の専用skydip (Step 2) から借用するので、この観測自体のGRAD状態
# (mini-skydip) は不要 -> despikeはSCAN_STATESのみでよい。

da_sub = reduce.despike(da, include=SCAN_STATES).sortby("time")
print(f"states in da_sub: {np.unique(da_sub.state.values)}, N = {da_sub.time.size}")

# %%
# --- Step 2: 近傍の専用skydipから2次校正係数を作る ---------------------------
# 手順 (skydip_pwv_vs_el_batch.py の process_skydip と同じロジック):
#   1. 専用skydipのSCANブロック (35-85度の広いスイープ) を取り、外部PWV参照を
#      初期値としてグローバルなPWVをfit (minimize + pwv_fit_func)
#   2. そのPWVとサンプルごとのelevationから eta_atm_los を計算し、モデルの
#      目標Tb (Tb_target = (1-eta_atm_los)*Tatm) を作る
#   3. 生df/fとTb_targetの2次多項式フィット (alpha*x^2 + a*x + b) をチャンネル
#      ごとに行う (pwv_pswsc_quad_calib.py のStep 8と同じ考え方)

da_ref = dc.load.dems(REFERENCE_SKYDIP_PATH)
if da_ref.long_name != "df/f":
    da_ref = dc.load.dems(REFERENCE_SKYDIP_PATH, data_scaling="df/f")
arr_ref = da_ref.to_numpy()
bad_ref = np.isnan(np.mean(arr_ref, axis=0)) | (np.nanstd(arr_ref, axis=0) == 0)
da_ref = da_ref[:, ~bad_ref]
del arr_ref
gc.collect()

da_ref_sub = reduce.despike(da_ref, include=["GRAD", "SCAN", "TRAN"]).sortby("time")
scan_ref = da_ref_sub[da_ref_sub.state == "SCAN"]
del da_ref, da_ref_sub
gc.collect()

eta_f, chan_toptica, f0_scan, freq_toptica = utils.load_filters()
eta_f = eta_f**2

Tatm_ref = np.nanmean(scan_ref.temperature.values)
freq_all = scan_ref.frequency.to_numpy()
chan_all = scan_ref.chan.to_numpy()
_, idx_in_toptica, idx_in_obs = np.intersect1d(
    chan_toptica, chan_all, assume_unique=True, return_indices=True
)
eta_f_sel = eta_f[:, idx_in_toptica]
chan_sel = chan_all[idx_in_obs]
freq_sel = freq_all[idx_in_obs]
scan_ref_sel = scan_ref[:, idx_in_obs]

gamma = np.nansum((eta_f_sel[:-1] + eta_f_sel[1:]) / 2 * np.diff(freq_toptica)[:, None], axis=0)
eta_fdf = (eta_f_sel[:-1] + eta_f_sel[1:]) / 2 * np.diff(freq_toptica)[:, None]
f_toptica_mid = (freq_toptica[1:] + freq_toptica[:-1]) / 2

el_ref = scan_ref_sel.lat.to_numpy() / 3600
csc_el_ref = 1 / np.sin(el_ref * np.pi / 180)

pwv0_ext = np.nanmean(get_pwv(utils.dt_to_seconds(scan_ref_sel)))
atm_interpol = get_eta_atm()

da_tb_ref = dc.convert.to_brightness(scan_ref_sel)
fit_args = (eta_fdf, f_toptica_mid, csc_el_ref, atm_interpol, gamma, Tatm_ref, da_tb_ref.to_numpy())
res = minimize(pwv_fit_func, pwv0_ext, args=fit_args, bounds=((0.1, 5.4),))
pwv_ref_global = float(np.squeeze(res.x))
print(f"reference skydip: external PWV0={pwv0_ext:.3f}mm, fitted global PWV={pwv_ref_global:.3f}mm, "
      f"el range=[{el_ref.min():.1f},{el_ref.max():.1f}]deg")

eta_atm_sample = atm_interpol(f_toptica_mid, pwv_ref_global).squeeze()[None, :] ** (csc_el_ref[:, None])
eta_atm_los = np.nansum(eta_fdf[None, :, :] * eta_atm_sample[:, :, None], axis=1) / gamma[None, :]  # (n_time, n_chan)
Tb_target = (1 - eta_atm_los) * Tatm_ref

x_raw_ref = scan_ref_sel.to_numpy()
n_ch = x_raw_ref.shape[1]
alpha2 = np.full(n_ch, np.nan)
a2 = np.full(n_ch, np.nan)
b2 = np.full(n_ch, np.nan)
R2 = np.full(n_ch, np.nan)
for ch in tqdm(range(n_ch), desc="quadratic calibration (reference skydip)"):
    x = x_raw_ref[:, ch].astype(np.float64)
    y = Tb_target[:, ch].astype(np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        continue
    p2 = np.polyfit(x[ok], y[ok], 2)
    alpha2[ch], a2[ch], b2[ch] = p2
    yhat = np.polyval(p2, x[ok])
    ss_res = np.sum((y[ok] - yhat) ** 2)
    ss_tot = np.sum((y[ok] - np.mean(y[ok])) ** 2)
    R2[ch] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

del da_tb_ref, scan_ref, scan_ref_sel, x_raw_ref, Tb_target, eta_atm_los, eta_atm_sample
gc.collect()

# %%
# --- Step 2.5: 校正品質によるチャンネル選別 ----------------------------------

keep = np.isfinite(alpha2) & np.isfinite(a2) & np.isfinite(b2) & (R2 > R2_THRES)
print(f"calibration: {keep.sum()} / {keep.size} channels kept (R2 > {R2_THRES}, "
      f"R2 median = {np.nanmedian(R2):.3f})")
if keep.sum() < 5:
    raise SystemExit("ERROR: 校正チャンネルがほぼゼロのため中断します。")
alpha2, a2, b2 = alpha2[keep], a2[keep], b2[keep]
idx_keep = idx_in_obs[keep]

# %%
# --- Step 3: 必要な配列だけnumpyに落とす (state ラベルも保持) ----------------

x_raw_target = da_sub[:, idx_keep].to_numpy()
tb = (alpha2[None, :] * x_raw_target**2 + a2[None, :] * x_raw_target + b2[None, :]).astype(np.float32)
del da, x_raw_target
gc.collect()
time = utils.dt_to_seconds(da_sub)
t_amb = da_sub.temperature.values.astype(np.float32)
csc_el = (1 / np.sin(da_sub.lat.values / 3600 / 180 * np.pi)).astype(np.float32)
chan = da_sub.chan.values[idx_keep]
freq = da_sub.frequency.values[idx_keep]
state = da_sub.state.values  # "ON" / "OFF" ラベル (chop-IC同定・較正に使う)

sel_fit = da_sub.beam.values == "A"  # ON+OFF両方 (元のOFF_NOD_ONLYは適用しない)
print(f"beam A samples (ON+OFF): {sel_fit.sum()} / {sel_fit.size} "
      f"(ON={np.sum((state == 'ON') & sel_fit)}, OFF={np.sum((state == 'OFF') & sel_fit)})")
del da_sub
gc.collect()

# %%
# --- 可視化: 校正済み生Tsky (beam A, 1チャンネル) ---------------------------

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(time[sel_fit], tb[sel_fit, CHAN_PLOT], ".", ms=1, color="k", alpha=0.5)
ax.set(xlabel="Time [sec]", ylabel="Tsky [K]", title=f"chan={CHAN_PLOT} (raw, beam A ON+OFF)")

# %%
# --- Step 3.2: tb の low-pass filter (beam A のみ) --------------------------

time_A = time[sel_fit]
tb_A = tb[sel_fit]

gap = np.diff(time_A)
lpf_period = 1.0 / LPF_CUTOFF_HZ
print(f"beam A time gaps: median={np.median(gap) * 1e3:.3f} ms, max={gap.max() * 1e3:.3f} ms "
      f"(LPF period = {lpf_period:.1f} s)")

dt0 = np.median(gap)
fs = 1.0 / dt0
t_grid = np.arange(time_A[0], time_A[-1] + dt0, dt0)

b_lpf, a_lpf = butter(LPF_ORDER, LPF_CUTOFF_HZ, btype="low", fs=fs)

tb_lpf = np.full_like(tb, np.nan)
tb_lpf_A = np.empty_like(tb_A)
for ch in tqdm(range(tb_A.shape[1]), desc="LPF per channel (beam A)"):
    ok = np.isfinite(tb_A[:, ch])
    y_grid = CubicSpline(time_A[ok], tb_A[ok, ch])(t_grid)
    y_grid_lpf = filtfilt(b_lpf, a_lpf, y_grid)
    tb_lpf_A[:, ch] = CubicSpline(t_grid, y_grid_lpf)(time_A)
tb_lpf[sel_fit] = tb_lpf_A

print(f"LPF done (beam A only): cutoff={LPF_CUTOFF_HZ} Hz, fs~{fs:.1f} Hz, order={LPF_ORDER}")

# %%
# --- 可視化: LPF前後・残差 (ON/OFF状態を色分け) ------------------------------

state_A = state[sel_fit]
on_mask = state_A == "ON"

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(time_A[on_mask], tb_A[on_mask, CHAN_PLOT], ".", ms=1, color="tab:red", alpha=0.5, label="ON")
axes[0].plot(time_A[~on_mask], tb_A[~on_mask, CHAN_PLOT], ".", ms=1, color="tab:blue", alpha=0.5, label="OFF")
axes[0].set_ylabel("Tsky [K]\n(raw, beam A)")
axes[0].legend(markerscale=8, fontsize=8)
axes[1].plot(time_A, tb_lpf_A[:, CHAN_PLOT], ".", ms=1, color="k", alpha=0.5)
axes[1].set_ylabel("Tsky [K]\n(LPF)")
axes[2].plot(time_A, tb_A[:, CHAN_PLOT] - tb_lpf_A[:, CHAN_PLOT], ".", ms=1, color="g", alpha=0.5)
axes[2].set_ylabel("Tsky [K]\n(residual)")
axes[-1].set(xlabel="Time [sec]")

# %%
# --- Step 3.5: 因子分析 + JADE (faica) --------------------------------------

valid = np.all(np.isfinite(tb_lpf), axis=1) & sel_fit
print(f"FA/JADE: using {valid.sum()} / {valid.size} time samples (finite rows only)")

# %%
# --- 可視化: 相関行列の固有値 (因子数の目安) --------------------------------

corr = np.corrcoef(tb_lpf[valid], rowvar=False)
eigvals = np.linalg.eigvalsh(corr)[::-1][:20]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(1, eigvals.size + 1), eigvals, ".-", ms=4, color="k")
ax.axhline(1, color="0.7", lw=0.5)
ax.set(xlabel="Component index", ylabel="Eigenvalue", yscale="log",
       title="tb_lpf correlation matrix eigenvalues")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# %%
fa = FactorAnalysis(n_components=N_FACTORS, random_state=0)
fa.fit(tb_lpf[valid])

Z, Q_white = whiten_fa(fa, tb_lpf[valid])

if N_JADE_SAMPLES is not None and Z.shape[1] > N_JADE_SAMPLES:
    rng = np.random.default_rng(0)
    sub = rng.choice(Z.shape[1], N_JADE_SAMPLES, replace=False)
    V = jade(Z[:, sub])
else:
    V = jade(Z)

sources = V @ Z  # (N_FACTORS, T_valid)
print(f"JADE done: {sources.shape[0]} independent components extracted")

# %%
# --- 可視化: 抽出した独立成分の時系列 (ON/OFF状態を色分け) -------------------

time_valid = time[valid]
state_valid = state[valid]
on_valid = state_valid == "ON"

n_show = min(30, N_FACTORS)
fig, axes = plt.subplots(n_show, 1, figsize=(12, 2.5 * n_show), sharex=True)
axes = np.atleast_1d(axes)
for i, ax in enumerate(axes):
    ax.plot(time_valid[on_valid], sources[i, on_valid], ".", ms=1, color="tab:red", alpha=0.5, label="ON")
    ax.plot(time_valid[~on_valid], sources[i, ~on_valid], ".", ms=1, color="tab:blue", alpha=0.5, label="OFF")
    ax.set_ylabel(f"IC{i + 1}")
axes[0].legend(markerscale=8, fontsize=8)
axes[-1].set(xlabel="Time [sec]")

# %%
# --- 可視化: 各ICの相関 (ON/OFFラベルとの相関係数) — 次段のIC選定の下調べ ---

onoff_indicator = on_valid.astype(float)
corr_with_chop = np.array([
    np.corrcoef(sources[i], onoff_indicator)[0, 1] for i in range(N_FACTORS)
])
print("correlation of each IC with the ON/OFF chop indicator:")
for i, c in enumerate(corr_with_chop):
    print(f"  IC{i + 1}: {c:+.3f}")

# %%
np.savez(
    f"step1_{obsid}.npz",
    sources=sources, time_valid=time_valid, state_valid=state_valid,
    V=V, fa_components=fa.components_, chan=chan, freq=freq,
    corr_with_chop=corr_with_chop,
)
print(f"saved: step1_{obsid}.npz")

# %%
