# %% Setup
from __future__ import annotations
import argparse, os, sys, re, json, hashlib, shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from itertools import combinations
import numpy as np
import pandas as pd
import pyreadstat
import matplotlib
from scipy import stats
from scipy.stats import t, norm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from statsmodels.stats.multitest import multipletests
base=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
data_name='Andrea_Modelo_Animal_TSH_Predito_RE_SPSS_New_241225.sav'
data_candidates=[folder/data_name for folder in (base,base.parent,base/'Arquivos Antigos')]
parser.add_argument('--data',type=Path)
parser.add_argument('--output-dir',type=Path,default=base/'results')
parser.add_argument('--images-dir',type=Path,default=base/'images')
parser.add_argument('--show',action='store_true')
spyder='spyder_kernels' in sys.modules
args=parser.parse_args([] if spyder else None)
SAV_PATH=args.data.resolve() if args.data is not None else next((path for path in data_candidates if path.is_file()),data_candidates[0])
results_path=args.output_dir.resolve();results_path.mkdir(parents=True,exist_ok=True)
images_path=args.images_dir.resolve()
images_path.mkdir(parents=True,exist_ok=True)
os.chdir(results_path)
DISPLAY_FIGURES=spyder or args.show
if spyder:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib','inline')
elif not DISPLAY_FIGURES:matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':110})
ALPHA=.05;ROBUST_TRIM=.2;WINSOR_P=.2;DPI=1200;SAVE_FIG=True;SAVE_EXCEL=True
GEE_COV_STRUCT='exchangeable'
groups=['Control','Hypothyroidism','LT4-Treated']
def present_figure(fig,caption):
    print(caption)
    if DISPLAY_FIGURES:plt.show()
    else:plt.close(fig)
def sheet(name):return re.sub(r'[\[\]:*?/\\]','_',name)[:31].rstrip()
def savefig(fig,name):
    fig.savefig(images_path/name,dpi=DPI,bbox_inches='tight');present_figure(fig,name)
def export_values(frame):
    return frame.map(lambda v:format(v,'.15g') if isinstance(v,(float,np.floating)) and np.isfinite(v) else None if isinstance(v,(float,np.floating)) else v)
def finish_excel(writer):
    from openpyxl.styles import Font,PatternFill,Alignment
    for ws in writer.sheets.values():
        ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
        ws.row_dimensions[1].height=58
        for cell in ws[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='24476A');cell.alignment=Alignment(wrap_text=True,vertical='center')
        for col in ws.columns:
            title=str(col[0].value);width=70 if 'explanation' in title.lower() or title in ['Notes','Description','Value_labels','CI_status','CI_method'] else 62 if ws.title=='Table_S1' else 27
            ws.column_dimensions[col[0].column_letter].width=width
        for row in ws.iter_rows(min_row=2):
            ws.row_dimensions[row[0].row].height=max(90 if ws.title in ['Table_S1','TSH_observations'] else 30,14*max((len(str(c.value or ''))//max(1,int(ws.column_dimensions[c.column_letter].width*.8))+1) for c in row))
            for cell in row:cell.alignment=Alignment(wrap_text=True,vertical='center')
def excel(path,tables):
    with pd.ExcelWriter(path,engine='openpyxl') as writer:
        for name,table in tables.items():export_values(table).to_excel(writer,sheet_name=sheet(name),index=False)
        finish_excel(writer)

# %% Shared statistical functions
# Legacy trimmed-Welch (source label: Yuen). Rank-deficient GEE Wald tests need review.
def _trim_and_winsor(x,trim):
    x=np.sort(np.asarray(x,dtype=float));x=x[np.isfinite(x)];g=int(np.floor(trim*len(x)))
    if not len(x):return x,x
    if g==0:return x,x.copy()
    z=x[g:-g];w=x.copy();w[:g]=x[g];w[-g:]=x[-g-1]
    return z,w

def xi_hat_from_winsor(a,b,trim=.2):
    a,aw=_trim_and_winsor(a,trim);b,bw=_trim_and_winsor(b,trim)
    if min(len(a),len(b))<2:return np.nan
    pooled=np.sqrt(((len(aw)-1)*np.var(aw,ddof=1)+(len(bw)-1)*np.var(bw,ddof=1))/(len(aw)+len(bw)-2))
    return (a.mean()-b.mean())/pooled if pooled>0 else np.nan

def yuen_pairwise(tmp,y,g,alpha=.05,trim=.2,minimum_n=4):
    rows=[];tmp=tmp[[g,y]].dropna();levels=sorted(tmp[g].astype(str).unique())
    for a,b in combinations(levels,2):
        x=tmp.loc[tmp[g].astype(str)==a,y].to_numpy(float);z=tmp.loc[tmp[g].astype(str)==b,y].to_numpy(float)
        if min(len(x),len(z))<minimum_n:continue
        xt,_=_trim_and_winsor(x,trim);zt,_=_trim_and_winsor(z,trim)
        if min(len(xt),len(zt))<2:continue
        v1=np.var(xt,ddof=1);v2=np.var(zt,ddof=1);n1=len(xt);n2=len(zt);se=np.sqrt(v1/n1+v2/n2)
        if not np.isfinite(se) or se<=0:continue
        stat=(xt.mean()-zt.mean())/se;df=(v1/n1+v2/n2)**2/(v1**2/(n1**2*(n1-1))+v2**2/(n2**2*(n2-1)))
        p=float(2*t.sf(abs(stat),df))
        rows.append(dict(group1=a,group2=b,t=stat,df=df,p_raw=p,xi_hat=xi_hat_from_winsor(x,z,trim),significant=p<alpha))
    return pd.DataFrame(rows)

def trimmed_means_global(tmp,y,g,trim=.2,minimum_n=4):
    tmp=tmp[[g,y]].dropna();levels=sorted(tmp[g].astype(str).unique());means=[];variances=[];sizes=[]
    for label in levels:
        x=tmp.loc[tmp[g].astype(str)==label,y].to_numpy(float)
        if len(x)<minimum_n:continue
        z,w=_trim_and_winsor(x,trim)
        if len(z)<2:continue
        means.append(z.mean());variances.append(np.var(w,ddof=1));sizes.append(len(z))
    k=len(means)
    if k<2:return np.nan,np.nan,len(tmp),len(levels)
    center=np.mean(means);num=sum(n*(v-center)**2 for n,v in zip(sizes,means));den=sum((n-1)*v for n,v in zip(sizes,variances))
    f=num/den if den>0 else np.nan;p=1-stats.f.cdf(f,k-1,sum(sizes)-k)
    return f,p,len(tmp),len(levels)

def run_oneway_pipeline(df_in,y,g='Group',alpha=.05,minimum_n=4):
    tmp=df_in[[g,y]].dropna();f,p,n,k=trimmed_means_global(tmp,y,g,ROBUST_TRIM,minimum_n)
    post=yuen_pairwise(tmp,y,g,alpha,ROBUST_TRIM,minimum_n)
    effect=float(post.xi_hat.abs().max()) if len(post) else np.nan
    return dict(ok=k>1,mode='Original trimmed pairwise',y_used=y,stat=f,p=p,effect_size=effect,effect_size_type='xi_hat_max_abs',posthoc='Original trimmed pairwise',posthoc_df=post,sig_pairs=[(r.group1,r.group2,r.p_raw) for r in post.itertuples() if r.significant],n=n,k=k)

def log_transform_safe(x):
    x = pd.to_numeric(x, errors='coerce')
    finite = x[np.isfinite(x)]
    if finite.empty:
        return x * np.nan
    min_val = float(finite.min())
    if min_val > 0:
        return np.log(x)
    positives = finite[finite > 0]
    if positives.empty:
        return x * np.nan
    offset = float(positives.min()) / 2.0
    return np.log(x + offset)

def _gee_cov_struct(name):
    name = (name or '').strip().lower()
    if name == 'independence':
        return Independence()
    return Exchangeable()

def gee_repeated_measures(df_long, y, g, tcol, subj, alpha=0.05):
    tmp = df_long[[subj, g, tcol, y]].copy()
    tmp[y] = pd.to_numeric(tmp[y], errors='coerce')
    tmp[tcol] = pd.to_numeric(tmp[tcol], errors='coerce')
    tmp = tmp.dropna(subset=[subj, g, tcol, y])
    if tmp.empty:
        return {'ok': False, 'reason': 'No data.'}
    tmp[g] = tmp[g].astype(str).astype('category')
    tmp[subj] = tmp[subj].astype(str)
    tmp[tcol] = tmp[tcol].astype(int)
    if tmp[g].nunique() < 2 or tmp[tcol].nunique() < 2:
        return {'ok': False, 'reason': 'Fewer than two levels in Group or Time.'}
    tmp['_logy'] = log_transform_safe(tmp[y])
    tmp = tmp.dropna(subset=['_logy'])
    if tmp.empty or tmp[g].nunique() < 2 or tmp[tcol].nunique() < 2:
        return {'ok': False, 'reason': 'Fewer than two levels after log transformation.'}
    tmp2 = tmp.rename(columns={g: 'Group', tcol: 'Time', subj: 'Subject'})
    cov_struct = _gee_cov_struct(GEE_COV_STRUCT)
    try:
        model = smf.gee(formula='_logy ~ C(Group) * C(Time)', groups='Subject', data=tmp2, family=Gaussian(), cov_struct=cov_struct)
        res = model.fit()
    except Exception as e:
        return {'ok': False, 'reason': f'GEE failed: {e}'}
    try:
        w = res.wald_test_terms(skip_single=False, scalar=True)
        wtab = getattr(w, 'table', None)
        if wtab is None:
            wtab = pd.DataFrame(w)
    except Exception:
        wtab = None

    def _wald_scalar(value):
        values = np.asarray(value, dtype=float)
        if values.size != 1:
            raise ValueError(f'Esperado um valor no teste de Wald; shape={values.shape}.')
        return float(values.item())

    def _term_p(term_contains):
        if wtab is None or not isinstance(wtab, pd.DataFrame):
            return np.nan
        idx = [i for i in wtab.index if term_contains == str(i)]
        if not idx:
            return np.nan
        row = wtab.loc[idx[0]]
        for col in ['pvalue', 'P>chi2', 'Pr(>chi2)', 'Pr>Chi2', 'p']:
            if col in wtab.columns:
                return _wald_scalar(row[col])
        nums = pd.to_numeric(row, errors='coerce')
        return float(nums.dropna().iloc[-1]) if np.isfinite(nums).any() else np.nan

    def _term_stat(term_contains):
        if wtab is None or not isinstance(wtab, pd.DataFrame):
            return np.nan
        idx = [i for i in wtab.index if term_contains == str(i)]
        if not idx:
            return np.nan
        row = wtab.loc[idx[0]]
        for col in ['chi2', 'statistic', 'Wald', 'Chi2']:
            if col in wtab.columns:
                return _wald_scalar(row[col])
        nums = pd.to_numeric(row, errors='coerce')
        return float(nums.dropna().iloc[0]) if np.isfinite(nums).any() else np.nan
    p_group = _term_p('C(Group)')
    p_time = _term_p('C(Time)')
    p_int = _term_p('C(Group):C(Time)')
    chi2_group = _term_stat('C(Group)')
    chi2_time = _term_stat('C(Time)')
    chi2_int = _term_stat('C(Group):C(Time)')
    try:
        yv = tmp2['_logy'].astype(float).values
        yhat = np.asarray(res.fittedvalues, dtype=float)
        sse = float(np.nansum((yv - yhat) ** 2))
        sst = float(np.nansum((yv - np.nanmean(yv)) ** 2))
        pseudo_r2 = float(1.0 - sse / sst) if np.isfinite(sse) and np.isfinite(sst) and (sst > 0) else np.nan
    except Exception:
        pseudo_r2 = np.nan
    model_data = getattr(res, 'model', None)
    design_info = None
    try:
        design_info = model_data.data.design_info if getattr(model_data, 'data', None) is not None else None
    except Exception:
        design_info = None
    return {'ok': True, 'mode': f'GEE robust ({GEE_COV_STRUCT})', 'n_obs': int(tmp2.shape[0]), 'n_subjects': int(tmp2['Subject'].nunique()), 'chi2_group': chi2_group, 'chi2_time': chi2_time, 'chi2_interaction': chi2_int, 'p_group': p_group, 'p_time': p_time, 'p_interaction': p_int, 'effect_size_type': 'pseudo_R2_log', 'pseudo_r2': pseudo_r2, 'res': res, 'design_info': design_info}

def gee_pairwise_by_time(gee_out, groups_order, times, alpha=0.05):
    out = {}
    if not isinstance(gee_out, dict) or not gee_out.get('ok', False):
        return out
    res = gee_out.get('res', None)
    design_info = gee_out.get('design_info', None)
    if res is None or design_info is None:
        return out
    cov = res.cov_params()
    params = res.params
    try:
        import patsy
    except Exception as e:
        raise ImportError('The patsy package is required for formula model contrasts.') from e

    def _row(g, tval):
        df_new = pd.DataFrame({'Group': [str(g)], 'Time': [int(tval)]})
        X = patsy.build_design_matrices([design_info], df_new, return_type='dataframe')[0]
        return np.asarray(X.iloc[0, :], dtype=float)
    times = np.asarray(times, dtype=int)
    groups_order = [str(g) for g in groups_order]
    for tt in sorted(np.unique(times).tolist()):
        all_pairs = []
        sig_pairs = []
        rows = {g: _row(g, tt) for g in groups_order}
        for i in range(len(groups_order)):
            for j in range(i + 1, len(groups_order)):
                g1, g2 = (groups_order[i], groups_order[j])
                c = rows[g1] - rows[g2]
                diff = float(np.dot(c, params))
                var = float(np.dot(c, np.dot(cov, c)))
                se = float(np.sqrt(var)) if np.isfinite(var) and var > 0 else np.nan
                z = float(diff / se) if np.isfinite(se) and se > 0 else np.nan
                p = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
                all_pairs.append((g1, g2, p, diff, se, z))
                if np.isfinite(p) and p < alpha:
                    sig_pairs.append((g1, g2, p))
        out[int(tt)] = {'ok': True, 'sig_pairs': sig_pairs, 'all_pairs': all_pairs}
    return out

def _gee_terms_table(gee_out, outcome_name):
    if not isinstance(gee_out, dict) or not gee_out.get('ok', False):
        return pd.DataFrame([{'Outcome': outcome_name, 'ok': False, 'reason': gee_out.get('reason', 'no result') if isinstance(gee_out, dict) else 'no result'}])
    res = gee_out.get('res', None)
    if res is None:
        return pd.DataFrame([{'Outcome': outcome_name, 'ok': False, 'reason': 'GEE result object missing'}])
    try:
        w = res.wald_test_terms(skip_single=False, scalar=True)
        wtab = getattr(w, 'table', None)
        if wtab is None:
            wtab = pd.DataFrame(w)
        wtab = wtab.copy()
        wtab.insert(0, 'Outcome', outcome_name)
        wtab.insert(1, 'Mode', gee_out.get('mode', ''))
        wtab.insert(2, 'n_obs', int(gee_out.get('n_obs', np.nan)) if np.isfinite(gee_out.get('n_obs', np.nan)) else np.nan)
        wtab.insert(3, 'n_subjects', int(gee_out.get('n_subjects', np.nan)) if np.isfinite(gee_out.get('n_subjects', np.nan)) else np.nan)
        wtab.reset_index(inplace=True)
        wtab.rename(columns={'index': 'Term'}, inplace=True)
    except Exception as e:
        wtab = pd.DataFrame([{'Outcome': outcome_name, 'ok': False, 'reason': f'wald_test_terms failed: {e}'}])
    return wtab

def _gee_params_table(gee_out, outcome_name):
    if not isinstance(gee_out, dict) or not gee_out.get('ok', False):
        return pd.DataFrame()
    res = gee_out.get('res', None)
    if res is None:
        return pd.DataFrame()
    try:
        params = pd.Series(res.params, name='coef').to_frame()
        se = pd.Series(res.bse, name='se').to_frame()
        z = pd.Series(res.params / res.bse, name='z').to_frame()
        p = pd.Series(2 * (1 - norm.cdf(np.abs(res.params / res.bse))), name='p').to_frame()
        out = pd.concat([params, se, z, p], axis=1).reset_index().rename(columns={'index': 'Parameter'})
        out.insert(0, 'Outcome', outcome_name)
        out.insert(1, 'Mode', gee_out.get('mode', ''))
        return out
    except Exception:
        return pd.DataFrame()

def cramers_v_from_chi2(chi2, n, r, c, bias_correct=True):
    if n <= 0 or r < 2 or c < 2 or (not np.isfinite(chi2)) or (chi2 < 0):
        return np.nan
    phi2 = chi2 / n
    if not bias_correct:
        return float(np.sqrt(phi2 / min(r - 1, c - 1)))
    phi2corr = max(0.0, phi2 - (c - 1) * (r - 1) / (n - 1)) if n > 1 else 0.0
    rcorr = r - (r - 1) ** 2 / (n - 1) if n > 1 else r
    ccorr = c - (c - 1) ** 2 / (n - 1) if n > 1 else c
    denom = min(rcorr - 1, ccorr - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2corr / denom))

def should_use_log_axis(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 6:
        return False
    if np.any(y <= 0):
        return False
    p05 = float(np.nanpercentile(y, 5))
    p95 = float(np.nanpercentile(y, 95))
    if not (np.isfinite(p05) and np.isfinite(p95)) or p05 <= 0:
        return False
    dyn = p95 / p05 if p05 > 0 else np.nan
    try:
        sk = float(stats.skew(y, bias=False, nan_policy='omit'))
    except Exception:
        sk = np.nan
    if np.isfinite(dyn) and dyn >= 50:
        return True
    if np.isfinite(sk) and abs(sk) >= 2.0:
        return True
    return False

def format_p_text(p):
    if not np.isfinite(p):
        return 'p=?'
    if p < 0.001:
        return 'p<0.001'
    return f'p={p:.3f}'

def sensitivity():
    from itertools import combinations
    from pathlib import Path
    SENS_PREFIX = 'Sensitivity_TSH'
    SENS_SEED = 261026
    SENS_PERMUTATIONS = 19999
    SENS_BOOTSTRAPS = 5000
    sens_rng = np.random.default_rng(SENS_SEED)
    sens_tsh = 'Imputed TSH (ng/mL)'
    sens_genes = ['Or5t5 Blood RE', 'Or10j5 Blood RE']
    sens_required = ['ID', 'qTSH_ngml', 'pTSH_ngml', 'cTSH_ngmL']
    if any((c not in df_raw for c in sens_required)):
        raise KeyError(f'Fields required for sensitivity analysis: {sens_required}')
    if len(target_pairs) != 27:
        raise ValueError('The directed testing family has changed. Review the protocol before running.')
    if not df_work.index.equals(df_raw.index):
        raise ValueError('Indices differ: review animal pairing.')
    sens_sources = pd.DataFrame({'ID': df_raw['ID'], 'Group': df[status_col].astype(str), 'Measured': pd.to_numeric(df_raw['qTSH_ngml'], errors='coerce'), 'ANN': pd.to_numeric(df_raw['pTSH_ngml'], errors='coerce'), 'Manuscript': pd.to_numeric(df_raw['cTSH_ngmL'], errors='coerce')}).replace([np.inf, -np.inf], np.nan)
    sens_measured = sens_sources['Measured'].notna()
    if sens_sources['ID'].duplicated().any() or sens_sources['ID'].isna().any():
        raise ValueError('Missing or duplicate IDs: review the experimental unit.')
    if not np.allclose(sens_sources.loc[sens_measured, 'Manuscript'], sens_sources.loc[sens_measured, 'Measured'], equal_nan=False):
        raise ValueError('Manuscript TSH does not preserve measured values.')
    print(f'\nSensitivity: {sens_measured.sum()} measured out of {len(sens_sources)} animals.')
    sens_observations = sens_sources.loc[sens_measured].copy()
    for gene in sens_genes:
        sens_observations[gene] = df_work.loc[sens_measured, gene]
    sens_observations['ANN_minus_measured'] = sens_observations['ANN'] - sens_observations['Measured']
    sens_desc_rows=[]
    for subset,mask in [('Available',np.ones(len(sens_sources),bool)),('Matched measured subset',sens_measured)]:
        for source in ['Manuscript','Measured','ANN']:
            for group in status_levels:
                values=sens_sources.loc[mask & (sens_sources.Group==str(group)),source].dropna()
                q1,median,q3=values.quantile([.25,.5,.75])
                sens_desc_rows.append(dict(Cohort=subset,Source=source,Group=str(group),n=len(values),Median=median,Q1=q1,Q3=q3,IQR=q3-q1,Minimum=values.min(),Maximum=values.max(),Range=values.max()-values.min(),Observations='; '.join(f'{x:g}' for x in values)))
    sens_descriptives=pd.DataFrame(sens_desc_rows)
    sens_orders = {source: block.sort_values('Median', kind='stable')['Group'].tolist() for source, block in sens_descriptives.loc[sens_descriptives.Cohort=='Available'].groupby('Source', sort=False)}
    sens_order_persists = sens_orders['Manuscript'] == sens_orders['Measured']
    sens_group_test = {'Method': 'Not computed', 'p': np.nan, 'Allocations': 0}
    sens_values = sens_sources.loc[sens_measured, 'Measured'].to_numpy()
    sens_labels = sens_sources.loc[sens_measured, 'Group'].to_numpy()
    sens_sizes = [int(np.sum(sens_labels == str(g))) for g in status_levels]
    if len(sens_values) == 9 and len(sens_sizes) == 3 and (min(sens_sizes) > 0):
        ranks = stats.rankdata(sens_values)
        tie_factor = stats.tiecorrect(ranks)

        def _sens_h(index_groups):
            if tie_factor == 0:
                return 0.0
            return (12 / (9 * 10) * sum((ranks[idx].sum() ** 2 / len(idx) for idx in index_groups)) - 30) / tie_factor
        observed_h = _sens_h([np.flatnonzero(sens_labels == str(g)) for g in status_levels])
        extreme, allocations = (0, 0)
        for first in combinations(range(9), sens_sizes[0]):
            remaining = sorted(set(range(9)) - set(first))
            for second in combinations(remaining, sens_sizes[1]):
                third = sorted(set(remaining) - set(second))
                h = _sens_h([list(first), list(second), third])
                extreme += h >= observed_h - 1e-12
                allocations += 1
        sens_group_test = dict(Method='Exact label permutation of Kruskal H (exploratory)', H=observed_h, p=extreme / allocations, Allocations=allocations)

    def _sens_coefficient(x, y, method, axis=-1):
        if method == 'Winsorized Pearson (20%)':
            low, high = np.quantile(x, [WINSOR_P, 1 - WINSOR_P], axis=axis, keepdims=True)
            x = np.clip(x, low, high)
            low, high = np.quantile(y, [WINSOR_P, 1 - WINSOR_P], axis=axis, keepdims=True)
            y = np.clip(y, low, high)
        else:
            x, y = (stats.rankdata(x, axis=axis), stats.rankdata(y, axis=axis))
        x = x - x.mean(axis=axis, keepdims=True)
        y = y - y.mean(axis=axis, keepdims=True)
        denominator = np.sqrt((x * x).sum(axis=axis) * (y * y).sum(axis=axis))
        return np.divide((x * y).sum(axis=axis), denominator, out=np.full(np.shape(denominator), np.nan), where=denominator > 0)
    sens_versions = [('Manuscript (all)', 'Manuscript', sens_sources.index), ('Measured only', 'Measured', sens_sources.index[sens_measured]), ('ANN (all)', 'ANN', sens_sources.index), ('ANN (measured subset)', 'ANN', sens_sources.index[sens_measured])]
    sens_families = []
    for version, source, indices in sens_versions:
        cohort = df_work.loc[indices].replace([np.inf, -np.inf], np.nan).copy()
        cohort[sens_tsh] = sens_sources.loc[indices, source]
        for method in ['Winsorized Pearson (20%)', 'Spearman (exploratory)']:
            rows = []
            for x_name, y_name in target_pairs:
                pair = cohort[[x_name, y_name]].dropna()
                x, y = pair.to_numpy(dtype=float).T
                n = len(pair)
                coefficient, p = (np.nan, np.nan)
                low, high, valid_boot = (np.nan, np.nan, 0)
                ci_note = 'Not requested for non-TSH pairs'
                is_tsh_target = sens_tsh in (x_name, y_name)
                if n >= 3:
                    coefficient = float(_sens_coefficient(x, y, method))
                    if np.isfinite(coefficient):
                        if method.startswith('Winsorized'):
                            original_pair = winsorized_pearson_pairwise(cohort, [x_name, y_name], WINSOR_P).iloc[0]
                            coefficient, p = (original_pair['r'], original_pair['p_value_raw'])
                        else:
                            xr, yr = (stats.rankdata(x), stats.rankdata(y))
                            xr, yr = (xr - xr.mean(), yr - yr.mean())
                            denom = np.sqrt(np.sum(xr * xr) * np.sum(yr * yr))
                            permutations = np.array([sens_rng.permutation(n) for _ in range(SENS_PERMUTATIONS)])
                            null = xr[permutations] @ yr / denom
                            p = (1 + np.sum(np.abs(null) >= abs(coefficient) - 1e-12)) / (SENS_PERMUTATIONS + 1)
                        if is_tsh_target:
                            ci_note = 'CI omitted: fewer than 6 pairs'
                            if n >= 6:
                                ix = sens_rng.integers(n, size=(SENS_BOOTSTRAPS, n))
                                boot = _sens_coefficient(x[ix], y[ix], method)
                                valid_boot = int(np.isfinite(boot).sum())
                                ci_note = 'CI omitted: >5% degenerate bootstrap samples'
                                if valid_boot >= 0.95 * SENS_BOOTSTRAPS:
                                    low, high = np.nanquantile(boot, [0.025, 0.975])
                                    ci_note = 'Exploratory paired percentile bootstrap 95%; small-n instability'
                if not np.isfinite(coefficient):
                    ci_note = 'Not estimable: <3 pairs or constant after transformation'
                rows.append(dict(Version=version, Method=method, Variable_X=x_name, Variable_Y=y_name, n=n, r=coefficient, p_value_raw=p, CI95_low=low, CI95_high=high, Valid_bootstraps=valid_boot, CI_note=ci_note))
            family = pd.DataFrame(rows)
            valid = np.isfinite(family['p_value_raw'])
            adjusted = multipletests(family['p_value_raw'].fillna(1), method='fdr_bh')[1]
            family['p_value_bh'] = np.where(valid, adjusted, np.nan)
            family['BH_family_size'] = len(target_pairs)
            family['significant_bh'] = family['p_value_bh'] < ALPHA
            sens_families.append(family)
    sens_correlations = pd.concat(sens_families, ignore_index=True)
    sens_tsh_results = sens_correlations.loc[(sens_correlations.Variable_X == sens_tsh) | (sens_correlations.Variable_Y == sens_tsh)].copy()
    sens_tsh_results['Gene'] = np.where(sens_tsh_results.Variable_X == sens_tsh, sens_tsh_results.Variable_Y, sens_tsh_results.Variable_X)

    def _sens_result(version, gene, method='Winsorized Pearson (20%)'):
        return sens_tsh_results.loc[(sens_tsh_results.Version == version) & (sens_tsh_results.Gene == gene) & (sens_tsh_results.Method == method)].iloc[0]

    def _sens_corr_text(row):
        if not np.isfinite(row.r):
            return f'Not estimable; n={row.n}'
        ci = f'; exploratory 95% CI [{row.CI95_low:.3f}, {row.CI95_high:.3f}]' if np.isfinite(row.CI95_low) else '; 95% CI unavailable'
        return f'r={row.r:.3f}; n={row.n}; p={row.p_value_raw:.4f}; BH p={row.p_value_bh:.4f}' + ci

    def _sens_distribution_text(source,group,matched=False):
        subset='Matched measured subset' if matched else 'Available'
        row=sens_descriptives.loc[(sens_descriptives.Source==source)&(sens_descriptives.Group==str(group))&(sens_descriptives.Cohort==subset)].iloc[0]
        return f'n={row.n}; median {row.Median:.2f}; Q1-Q3 [{row.Q1:.2f}, {row.Q3:.2f}]; min-max [{row.Minimum:.2f}, {row.Maximum:.2f}]'
    sens_review=pd.DataFrame([dict(Gene=g,Measured_r=_sens_result('Measured only',g).r,ANN_r=_sens_result('ANN (all)',g).r,Manuscript_r=_sens_result('Manuscript (all)',g).r,ANN_matched_r=_sens_result('ANN (measured subset)',g).r,Measured_BH=_sens_result('Measured only',g).significant_bh,ANN_BH=_sens_result('ANN (all)',g).significant_bh,Manuscript_BH=_sens_result('Manuscript (all)',g).significant_bh) for g in sens_genes])
    tsh_labels={'Measured':'Experimentally measured TSH (ng/mL)','ANN':'ANN-predicted TSH (ng/mL)','Manuscript':'Manuscript TSH (ng/mL)'}
    sens_s1_rows=[]
    for matched in [False,True]:
        tag='Matched measured subset' if matched else 'Available samples'
        versions={'Measured':'Measured only','ANN':'ANN (measured subset)' if matched else 'ANN (all)','Manuscript':'Measured only' if matched else 'Manuscript (all)'}
        for group in status_levels:sens_s1_rows.append({'Analysis':f'{tag}: {group}',**{label:_sens_distribution_text(key,group,matched) for key,label in tsh_labels.items()}})
        for gene in sens_genes:sens_s1_rows.append({'Analysis':f'{tag}: TSH x {gene}',**{label:_sens_corr_text(_sens_result(versions[key],gene)) for key,label in tsh_labels.items()}})
    sens_table_s1=pd.DataFrame(sens_s1_rows)
    sens_notes = ['Exploratory measured-TSH sensitivity; paired bootstrap B=5000; BH over 27 pairs; seed 261026.']
    for suffix, table in [('Table_S1', sens_table_s1), ('observations', sens_observations), ('descriptives', sens_descriptives), ('all_families', sens_correlations), ('TSH_correlations', sens_tsh_results), ('review', sens_review)]:
        table.to_csv(f'{SENS_PREFIX}_{suffix}.csv', index=False, sep=';', encoding='utf-8-sig')
    sens_md = [sens_notes[0], '', '| ' + ' | '.join(sens_table_s1.columns) + ' |', '| --- | --- | --- |']
    sens_md += ['| ' + ' | '.join(map(str, row)) + ' |' for row in sens_table_s1.itertuples(index=False, name=None)]
    sens_md += ['', *[f'- {note}' for note in sens_notes[1:]]]
    Path(f'{SENS_PREFIX}_Table_S1.md').write_text('\n'.join(sens_md), encoding='utf-8')
    print(sens_table_s1.to_string(index=False))
    print(sens_review.to_string(index=False))

    print('Exploratory global permutation:', sens_group_test)
    sens_fig, sens_ax = plt.subplots(figsize=(7.5, 4.5), dpi=150, constrained_layout=True)
    for i, group in enumerate(status_levels):
        block = sens_observations.loc[sens_observations.Group == str(group)]
        values = block.Measured.to_numpy()
        positions = i + np.linspace(-0.12, 0.12, len(values))
        sens_ax.scatter(positions, values, s=35, color='0.15', zorder=3)
        for pos, (_, obs) in zip(positions, block.iterrows()):
            sens_ax.annotate(f'ID {obs.ID:g}', (pos, obs.Measured), xytext=(4, 4), textcoords='offset points', fontsize=7)
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        sens_ax.vlines(i, values.min(), values.max(), color='0.6', lw=1)
        sens_ax.vlines(i, q1, q3, color='steelblue', lw=7, alpha=0.45)
        sens_ax.hlines(med, i - 0.2, i + 0.2, color='steelblue', lw=2)
    sens_ax.set_xticks(range(len(status_levels)), labels=[f'{g} (n={sum(sens_observations.Group == str(g))})' for g in status_levels])
    sens_ax.set_ylabel('Experimentally measured TSH (ng/mL)')
    sens_ax.set_title('Measured TSH: observations, median, IQR and range', fontsize=11)
    sens_ax.spines[['top', 'right']].set_visible(False)
    if SAVE_FIG:
        sens_fig.savefig(images_path/f'{SENS_PREFIX}_Measured_distribution.png', dpi=DPI)
    present_figure(sens_fig, 'Supplement: measured TSH distribution')
    return locals()


# %% Load data and longitudinal analyses
if not SAV_PATH.is_file():
    searched=[SAV_PATH] if args.data is not None else data_candidates
    raise FileNotFoundError('SPSS input not found. Set the input in Setup or pass --data. Checked: ' + ', '.join(map(str,searched)))
df_raw,meta=pyreadstat.read_sav(str(SAV_PATH),apply_value_formats=False)
data=df_raw.copy();data['Group']=data['Status'].map(meta.variable_value_labels['Status'])
longitudinal={};models={};contrasts={};timepost=[]
for outcome,prefix in [('Weight','Peso_DG'),('BloodPressure','PAS_DG'),('BloodGlucose','Glicemia_DG')]:
    cols=[c for c in data if c.startswith(prefix)]
    long=data[['ID','Group']+cols].melt(id_vars=['ID','Group'],var_name='Variable',value_name=outcome)
    long['GD']=long.Variable.str.extract(r'(\d+)$').astype(int)
    longitudinal[outcome]=long
    models[outcome]=gee_repeated_measures(long,outcome,'Group','GD','ID')
    contrasts[outcome]=gee_pairwise_by_time(models[outcome],groups,np.sort(long.GD.unique()))
    for day,block in long.groupby('GD'):
        post=run_oneway_pipeline(block,outcome)['posthoc_df'].copy()
        post.insert(0,'PosthocMethod','Yuen (raw p-values) + xi_hat');post.insert(0,'GD',int(day));post.insert(0,'Outcome',outcome)
        timepost.append(post)
endpoint_specs=[('pH','Urinary pH'),('Proteinuria_imageJ','Proteinuria'),('Filhotes','Pups')]
genes=[('Or5t5_sangue','Or5t5 Blood'),('Or5t5_rim','Or5t5 Kidney'),('Or5t5_placenta','Or5t5 Placenta'),('Or10j5_sangue','Or10j5 Blood'),('Or10j5_rim','Or10j5 Kidney'),('Or10j5_placenta','Or10j5 Placenta'),('AT1_sangue','Agtr1a Blood'),('AT1_ve','Agtr1a Heart'),('AT1_placenta','Agtr1a Placenta'),('AT1_rim','Agtr1a Kidney'),('Mor_placenta','Oprm1'),('Pgf_placenta','Pgf')]
endpoints={}
for field,label in endpoint_specs:
    endpoints[field]=run_oneway_pipeline(data,field)
def significance(ax,post,blocks,*,p_label_gap=0,group_colors=None):
    sig=post.loc[post.significant]
    if sig.empty:return
    log=ax.get_yscale()=='log';top=max(float(np.max(x)) for x in blocks if len(x));bottom=min(float(np.min(x)) for x in blocks if len(x))
    # Leave room between stacked brackets for the offset p-value labels.
    step=(top-bottom)*(.17 if p_label_gap else .13) or 1
    for j,(_,row) in enumerate(sig.iterrows()):
        a,b=groups.index(row.group1),groups.index(row.group2)
        y=top*(1.45**(j+1)) if log else top+step*(j+1)
        bracket_top=y*1.01 if log else y+step*.1
        bracket_color = group_colors[a] if group_colors is not None else '.25'
        ax.plot([a,a,b,b],[y,bracket_top,bracket_top,y],color=bracket_color,lw=.7)
        label_y=bracket_top if p_label_gap else y
        ax.annotate(format_p_text(row.p_raw),((a+b)/2,label_y),
                    xytext=(0,p_label_gap),textcoords='offset points',
                    ha='center',va='bottom',fontsize=7,annotation_clip=False)
    ax.set_ylim(top=y*(1.3 if log else 1.15))
def boxes(ax,field,title,res,*,p_label_gap=0,group_colors=None):
    blocks=[data.loc[data.Group==g,field].dropna().to_numpy() for g in groups]
    if should_use_log_axis(np.concatenate([x for x in blocks if len(x)])) and all((x>0).all() for x in blocks):ax.set_yscale('log')
    ax.boxplot(blocks,positions=range(3),tick_labels=groups)
    for j,x in enumerate(blocks):
        color = group_colors[j] if group_colors is not None else '.3'
        ax.scatter(j+np.random.default_rng(123).normal(0,.04,len(x)),x,s=9,color=color,zorder=3)
    ax.set_title(title,loc='left')
    ax.tick_params(axis='x',labelrotation=0 if group_colors is not None else 25)
    significance(ax,res['posthoc_df'],blocks,p_label_gap=p_label_gap,group_colors=group_colors)

# %% Figure 1: physiological parameters
def comparison_bracket(ax,left,right,top,depth,p_value):
    ax.plot([left,left,right,right],[top-depth,top,top,top-depth],color='.3',lw=.9)
    ax.annotate(format_p_text(p_value),((left+right)/2,top),xytext=(0,3.5),
                textcoords='offset points',ha='center',va='bottom',fontsize=6)

def endpoint_bars(ax,field,panel_letter,ylabel):
    blocks=[data.loc[data.Group==g,field].dropna() for g in groups]
    means=np.array([x.mean() for x in blocks])
    sem=np.array([stats.sem(x) for x in blocks])
    ax.bar(range(len(groups)),means,yerr=sem,color=['.2','.5','.8'],capsize=2)
    ax.set_xticks(range(len(groups)),groups)
    ax.set_title(panel_letter,loc='left',weight='bold')
    ax.set_ylabel(ylabel+' (mean ± SEM)')
    peak=np.nanmax(means+np.nan_to_num(sem,nan=0))
    top=peak
    posthoc=endpoints[field]['posthoc_df']
    for _,row in posthoc.loc[posthoc.significant].iterrows():
        top+=peak*.12
        comparison_bracket(ax,groups.index(row.group1),groups.index(row.group2),top,peak*.028,row.p_raw)
    ax.set_ylim(top=top+peak*.13)

fig=plt.figure(figsize=(10,9));gs=fig.add_gridspec(3,2,hspace=.5,wspace=.3)
for index,(outcome,ylabel) in enumerate([('Weight','Weight (g)'),('BloodPressure','Blood pressure (mmHg)'),('BloodGlucose','Blood glucose (mg/dL)')]):
    ax=fig.add_subplot(gs[0,:] if index==0 else gs[1,index-1]);long=longitudinal[outcome];days=sorted(long.GD.unique());width=.25
    bar_tops=np.zeros(len(days))
    for j,g in enumerate(groups):
        values=long.loc[long.Group==g].groupby('GD')[outcome]
        means=values.mean().reindex(days)
        sem=values.sem().reindex(days)
        ax.bar(np.arange(len(days))+(j-1)*width,means,width,yerr=sem,color=['.2','.5','.8'][j],label=g,capsize=2)
        bar_tops=np.fmax(bar_tops,means.to_numpy()+sem.fillna(0).to_numpy())
    # Use one vertical scale per panel to separate labels from bars and each other.
    peak=bar_tops.max()
    annotation_top=peak
    for k,day in enumerate(days):
        pairs=contrasts[outcome][day].get('sig_pairs',[])
        # Draw shorter brackets first, breaking ties from right to left.
        positions=[(sorted((groups.index(g1),groups.index(g2))),p) for g1,g2,p in pairs]
        positions.sort(key=lambda pair:(pair[0][1]-pair[0][0],-pair[0][0]))
        for level,((left,right),p) in enumerate(positions):
            y=bar_tops[k]+peak*(.07+.18*level)
            a=k+(left-1)*width
            b=k+(right-1)*width
            comparison_bracket(ax,a,b,y,peak*.028,p)
            annotation_top=max(annotation_top,y)
    upper_limit=annotation_top+.13*peak
    if index==0:
        ax.legend(loc='upper right',fontsize=7)
        upper_limit=max(upper_limit,peak/.70)
    ax.set_ylim(top=upper_limit)
    ax.set_xticks(range(len(days)),days);ax.set_ylabel(ylabel+' (mean ± SEM)');ax.set_title('ABC'[index],loc='left',weight='bold')
endpoint_bars(fig.add_subplot(gs[2,0]),'pH','D','Urinary pH')
endpoint_bars(fig.add_subplot(gs[2,1]),'Proteinuria_imageJ','E','Proteinuria')
savefig(fig,'Figure_1_Physiological_Parameters.png')

# %% Figure 2 and thyroid statistics
hormones={f:run_oneway_pipeline(data,f,minimum_n=5) for f in ['cTSH_ngmL','T4','T3']}
ba=df_raw[['qTSH_ngml','pTSH_ngml']].dropna();mean=ba.mean(axis=1);difference=ba.iloc[:,0]-ba.iloc[:,1]
bias=difference.mean();sd=difference.std(ddof=1);r,p=stats.pearsonr(ba.iloc[:,0],ba.iloc[:,1])
agreement=pd.DataFrame([dict(n=len(ba),Pearson_r=r,R_squared=r*r,p_raw=p,bias_measured_minus_predicted=bias,SD_difference=sd,LoA_lower=bias-1.96*sd,LoA_upper=bias+1.96*sd)])
fig,axes=plt.subplots(2,2,figsize=(9.3,6.6),layout='constrained');ax=axes[0,0]
thyroid_colors = ('tab:blue', 'tab:orange', 'tab:green')
ax.scatter(mean, difference, s=12, color=thyroid_colors[0])
ax.axhline(bias, color=thyroid_colors[0])
ax.axhline(bias + 1.96 * sd, ls='--', color=thyroid_colors[0])
ax.axhline(bias - 1.96 * sd, ls='--', color=thyroid_colors[0])
ax.set(xlabel='Mean of TSH (ng/mL)', ylabel='Difference (First - Second) (ng/mL)')
ax.set_title(r'$\mathbf{A}$ Bland–Altman - Quantified TSH vs Predict TSH', loc='left')
ax.text(.02, .91,
        f'Mean diff = {bias:.2f}\nLoA (+1.96 SD) = {bias + 1.96 * sd:.2f}\n'
        f'LoA (−1.96 SD) = {bias - 1.96 * sd:.1f}',
        transform=ax.transAxes, va='top', fontsize=8)
thyroid_panels = [
    ('B', 'cTSH_ngmL', 'Imputed TSH (ng/mL)', 'Imputed TSH (ng/mL)'),
    ('C', 'T4', 'Total T4', 'Total T4 (µg/dL)'),
    ('D', 'T3', 'Total T3', 'Total T3 (ng/dL)'),
]
for ax, (letter, field, title, ylabel) in zip(axes.flat[1:], thyroid_panels):
    boxes(ax, field, rf'$\mathbf{{{letter}}}$ {title}', hormones[field],
          p_label_gap=6, group_colors=thyroid_colors)
    ax.set(xlabel='Status', ylabel=ylabel)
with matplotlib.rc_context({'axes.unicode_minus': True}):
    savefig(fig,'Figure_2_Thyroid_Hormones_Agreement.png')
def describe(field,label):
    return pd.DataFrame([dict(Outcome=label,Group=g,n_total=len(x),Mean=x.mean(),trimmed_mean=stats.trim_mean(x,.2),median=np.median(x),q1=np.quantile(x,.25),q3=np.quantile(x,.75)) for g in groups if len(x:=data.loc[data.Group==g,field].dropna())])
hormone_tables={'agreement':agreement};hdesc=[];hinf=[]
for field,label in [('cTSH_ngmL','Imputed TSH (ng/mL)'),('T4','Total T4 (µg/dL)'),('T3','Total T3 (ng/dL)')]:
    res=hormones[field];hdesc.append(describe(field,label));hinf.append(dict(Outcome=label,F_robust=res['stat'],p_global=res['p'],EffectSize=res['effect_size'],n_total=res['n'],k_groups=res['k']))
    hormone_tables['posthoc_'+label]=res['posthoc_df']
hormone_tables.update(inferential=pd.DataFrame(hinf),descriptive=pd.concat(hdesc,ignore_index=True));hormone_tables['all_results']=pd.concat([hormone_tables['inferential'],hormone_tables['descriptive']],ignore_index=True)
excel('Table_1_Thyroid_Hormone_Statistics.xlsx',hormone_tables)

# %% Figure 3: tissue gene expression
for field,label in genes:endpoints[field]=run_oneway_pipeline(data,field)
fig,axes=plt.subplots(3,4,figsize=(11,8),layout='constrained')
for i,(field,label) in enumerate(genes):boxes(axes.flat[i],field,chr(65+i)+'  '+label,endpoints[field])
savefig(fig,'Figure_3_Gene_Expression.png')

# %% Tables 1 and 2: inference, descriptives and nominal outcomes
tables={};rows=[];pairrows=[];terms=[];params=[]
for outcome,res in models.items():
    for effect,key in [('Group','group'),('Time','time'),('Group×Time','interaction')]:rows.append(dict(Outcome=outcome+' (repeated)' if outcome=='Weight' else {'BloodPressure':'Blood Pressure (repeated)','BloodGlucose':'Blood Glucose (repeated)'}[outcome],Design='Repeated',Test='GEE Wald',Effect=effect,Stat=res['chi2_'+key],p=res['p_'+key],EffectSize=res['pseudo_r2'],EffectSizeType='pseudo_R2_log',Notes=res['mode']))
    for day,block in contrasts[outcome].items():
        for g1,g2,p,diff,se,z in block.get('all_pairs',[]):pairrows.append(dict(Outcome=outcome,Time=int(day),group1=g1,group2=g2,p=p,diff=diff,se=se,z=z,significant=p<ALPHA))
    terms.append(_gee_terms_table(res,outcome));params.append(_gee_params_table(res,outcome))
for field,label in endpoint_specs+[(f,l+' Relative Expression') for f,l in genes]:
    res=endpoints[field];rows.append(dict(Outcome=label,Design='1-way',Test=res['mode'],Effect='Group',Stat=res['stat'],p=res['p'],EffectSize=res['effect_size'],EffectSizeType=res['effect_size_type'],Notes=res['posthoc']))
    name={'pH':'urinary_ph','Proteinuria_imageJ':'proteinuria','Filhotes':'pups'}.get(field,label);tables['posthoc_'+name]=res['posthoc_df']
pairs=pd.DataFrame(pairrows)
tables.update(inferential_effectsize=pd.DataFrame(rows),gee_effects_summary=pd.DataFrame([dict(Outcome=k,**{n:v for n,v in r.items() if n not in ['res','design_info']}) for k,r in models.items()]),gee_pairwise_by_time=pairs,gee_terms_wald=pd.concat(terms,ignore_index=True),gee_params=pd.concat(params,ignore_index=True),posthoc_timepoints=pd.concat(timepost,ignore_index=True))
wide=pairs.assign(Comparison=pairs.group1+' vs '+pairs.group2).pivot_table(index=['Outcome','Time'],columns='Comparison',values='p').reset_index();tables['gee_pairwise_by_time_wide']=wide
cs=[];ct=[];ce=[]
for field,label in [('Cetona','Ketone'),('Glicose','Glucose'),('Proteina','Protein'),('Sangue','Blood'),('Nitriro','Nitrite'),('Leucocitos','Leukocytes')]:
    observed=pd.crosstab(data.Group,data[field]).reindex(groups,fill_value=0)
    if min(observed.shape)<2:cs.append(dict(ok=False,Outcome='Group',Factor=field,reason='Only one level',n=int(observed.to_numpy().sum())));continue
    chi,p,dof,expected=stats.chi2_contingency(observed,correction=False);n=observed.to_numpy().sum()
    cs.append(dict(ok=True,Outcome='Group',Factor=field,chi2=chi,p_chi2=p,dof=dof,n=n,n_cells=expected.size,pct_expected_lt5=100*np.mean(expected<5),pct_expected_lt1=100*np.mean(expected<1),cramers_v=cramers_v_from_chi2(chi,n,*observed.shape)))
    for j,level in enumerate(observed.columns):
        for i,g in enumerate(groups):ct.append(dict(Factor=field,Status=g,FactorLevel=level,Count=observed.iloc[i,j]));ce.append(dict(Factor=field,Status=g,FactorLevel=level,Expected=expected[i,j]))
tables.update(chi_square_summary=pd.DataFrame(cs),chi_square_table=pd.DataFrame(ct),chi_square_expected=pd.DataFrame(ce),endpoint_descriptives=pd.concat([describe(f,l) for f,l in endpoint_specs+[(f,l+' Relative Expression') for f,l in genes]],ignore_index=True))
tables['gee_descriptives']=pd.concat([v.groupby(['Group','GD'])[k].agg(n='count',Mean='mean',Q1=lambda x:x.quantile(.25),Median='median',Q3=lambda x:x.quantile(.75)).reset_index().assign(Outcome=k) for k,v in longitudinal.items()],ignore_index=True)
for name in ['chi_square_summary','chi_square_table','chi_square_expected']:tables[name]['Factor']=tables[name].Factor.replace({'Nitriro':'Nitrite'})
excel('Tables_1_2_Physiology_Gene_Statistics.xlsx',tables)

# %% Figure 4: prepare correlations and BH families
FIELD_TO_LABEL = {'ID': 'ID', 'Protocolo': 'Protocol', 'Status': 'Thyroid Status', 'Peso_DG0': 'Weight 0', 'Peso_DG1': 'Weight 1', 'Peso_DG2': 'Weight 2', 'Peso_DG3': 'Weight 3', 'Peso_DG4': 'Weight 4', 'Peso_DG5': 'Weight 5', 'Peso_DG6': 'Weight 6', 'Peso_DG7': 'Weight 7', 'Peso_DG8': 'Weight 8', 'Peso_DG9': 'Weight 9', 'Peso_DG10': 'Weight 10', 'Peso_DG11': 'Weight 11', 'Peso_DG12': 'Weight 12', 'Peso_DG13': 'Weight 13', 'Peso_DG14': 'Weight 14', 'PAS_DG0': 'Blood Pressure GD0', 'PAS_DG6': 'Blood Pressure GD6', 'PAS_DG12': 'Blood Pressure GD12', 'qTSH_ngml': 'Quantified TSH (ng/mL)', 'pTSH_ngml': 'Predict TSH (ng/mL)', 'cTSH_ngmL': 'Imputed TSH (ng/mL)', 'T4': 'Total T4 (µg/dL)', 'T3': 'Total T3 (ng/dL)', 'Glicemia_DG0': 'Blood Glucose GD0', 'Glicemia_DG6': 'Blood Glucose GD6', 'Glicemia_DG12': 'Blood Glucose GD12', 'Cetona': 'Ketone', 'Glicose': 'Glucose', 'Proteina': 'Protein', 'Sangue': 'Blood', 'Nitriro': 'Nitrite', 'Leucocitos': 'Leukocytes', 'pH': 'pH', 'Proteinuria_imageJ': 'Proteinuria', 'Filhotes': 'Puppies', 'Sangue_A': 'Blood (Alt.)', 'Rim': 'Kidney', 'Placenta': 'Placenta', 'VE': 'Heart - VE', 'AT1_sangue': 'Agtr1a Blood RE', 'AT1_ve': 'Agtr1a Heart RE', 'AT1_placenta': 'Agtr1a Placenta RE', 'AT1_rim': 'Agtr1a Kidney RE', 'Or5t5_sangue': 'Or5t5 Blood RE', 'Or5t5_rim': 'Or5t5 Kidney RE', 'Or5t5_placenta': 'Or5t5 Placenta RE', 'Or10j5_sangue': 'Or10j5 Blood RE', 'Or10j5_rim': 'Or10j5 Kidney RE', 'Or10j5_placenta': 'Or10j5 Placenta RE', 'Mor_placenta': 'Oprm1 Placenta RE', 'Pgf_placenta': 'Pgf Placenta RE'}
def format_plot_label(label):
    clean = re.sub('\\s+RE$', '', str(label)).strip()
    match = re.match('^(Agtr1a|Or5t5|Or10j5|Oprm1|Pgf)(.*)$', clean)
    if not match:
        return clean
    gene, suffix = match.groups()
    return f'$\\it{{{gene}}}${suffix}'
def winsorize_series(s, p=0.2):
    s = pd.to_numeric(s, errors='coerce')
    x = s.dropna()
    if x.shape[0] < 3:
        return s
    lo = float(x.quantile(p, interpolation='linear'))
    hi = float(x.quantile(1.0 - p, interpolation='linear'))
    return s.clip(lower=lo, upper=hi)
def reorder_agtr1a_after_or10j5_placenta_before_oprm1(cols):
    cols = [str(c) for c in cols]
    agtr_order = ['Agtr1a Blood RE', 'Agtr1a Heart RE', 'Agtr1a Placenta RE', 'Agtr1a Kidney RE']
    agtr_present = [c for c in agtr_order if c in cols]
    if not agtr_present:
        return cols
    anchor_after = 'Or10j5 Placenta RE'
    anchor_before = 'Oprm1 RE'
    base = [c for c in cols if c not in set(agtr_present)]
    if anchor_after in base:
        insert_idx = base.index(anchor_after) + 1
    elif anchor_before in base:
        insert_idx = base.index(anchor_before)
    else:
        insert_idx = len(base)
    return base[:insert_idx] + agtr_present + base[insert_idx:]
df=df_raw.rename(columns=FIELD_TO_LABEL).copy();status_col='Thyroid Status'
df[status_col]=df_raw.Status.map(meta.variable_value_labels['Status']).astype('category');status_levels=list(df[status_col].cat.categories)
fields=['Peso_DG14','PAS_DG12','Glicemia_DG12','cTSH_ngmL','T4','T3','pH','Proteinuria_imageJ','Filhotes']+[f for f,_ in genes]
ordered=fields[:3]+[c for c in df_raw if c in fields and c not in fields[:3]]
df_work=df[[status_col]+[FIELD_TO_LABEL.get(c,c) for c in ordered]].copy()
df_work.rename(columns={'Weight 14':'Last Weight','Blood Pressure GD12':'Last Blood Preassure','Blood Glucose GD12':'Last Glicemia'},inplace=True)
for c in df_work:
    if c!=status_col:df_work[c]=pd.to_numeric(df_work[c],errors='coerce')
numeric_cols=reorder_agtr1a_after_or10j5_placenta_before_oprm1([c for c in df_work if c!=status_col])
circulating_or_cols=['Or5t5 Blood RE','Or10j5 Blood RE']
rna_target_cols=['Imputed TSH (ng/mL)','Total T4 (µg/dL)','Total T3 (ng/dL)','Or5t5 Blood RE','Or10j5 Blood RE','Agtr1a Blood RE','Agtr1a Heart RE','Agtr1a Placenta RE','Agtr1a Kidney RE','Or5t5 Kidney RE','Or5t5 Placenta RE','Or10j5 Kidney RE','Or10j5 Placenta RE','Oprm1 Placenta RE','Pgf Placenta RE']
target_pairs=[];seen=set()
for x in circulating_or_cols:
    for y in rna_target_cols:
        key=tuple(sorted([x,y]))
        if x!=y and key not in seen:target_pairs.append((x,y));seen.add(key)
def winsorized_pearson_pairwise(df_in, cols, winsor_p=0.2):
    rows = []
    cols = [c for c in cols if c in df_in.columns]
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            x, y = (cols[i], cols[j])
            pair = df_in[[x, y]].dropna()
            n = int(pair.shape[0])
            if n < 3:
                rows.append({'Variable_X': x, 'Variable_Y': y, 'Method': f'Winsorized Pearson ({int(winsor_p * 100)}%)', 'Winsor_p': winsor_p, 'n': n, 'r': np.nan, 'p_value_raw': np.nan})
                continue
            xw = winsorize_series(pair[x], p=winsor_p)
            yw = winsorize_series(pair[y], p=winsor_p)
            xw = pd.to_numeric(xw, errors='coerce')
            yw = pd.to_numeric(yw, errors='coerce')
            mask = np.isfinite(xw.values) & np.isfinite(yw.values)
            if mask.sum() < 3:
                rows.append({'Variable_X': x, 'Variable_Y': y, 'Method': f'Winsorized Pearson ({int(winsor_p * 100)}%)', 'Winsor_p': winsor_p, 'n': int(mask.sum()), 'r': np.nan, 'p_value_raw': np.nan})
                continue
            r, p = stats.pearsonr(xw.values[mask].astype(float), yw.values[mask].astype(float))
            rows.append({'Variable_X': x, 'Variable_Y': y, 'Method': f'Winsorized Pearson ({int(winsor_p * 100)}%)', 'Winsor_p': winsor_p, 'n': int(mask.sum()), 'r': float(r) if np.isfinite(r) else np.nan, 'p_value_raw': float(p) if np.isfinite(p) else np.nan})
    return pd.DataFrame(rows)

def winsorized_pearson_selected_pairs(df_in, pairs, winsor_p=0.2):
    parts = [winsorized_pearson_pairwise(df_in, cols=[x, y], winsor_p=winsor_p) for x, y in pairs]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def add_fdr_adjustments(corr_df,alpha=.05):
    out=corr_df.copy();p=pd.to_numeric(out.p_value_raw,errors='coerce');valid=np.isfinite(p)
    out['p_value_bh']=np.nan
    if valid.any():out.loc[valid,'p_value_bh']=multipletests(p[valid],method='fdr_bh')[1]
    out['BH_family_size']=int(valid.sum())
    out['significant_raw']=p<alpha;out['significant_bh']=out.p_value_bh<alpha
    return out
corr_global_df = winsorized_pearson_selected_pairs(df_work, pairs=target_pairs, winsor_p=WINSOR_P)
corr_global_df = add_fdr_adjustments(corr_global_df, alpha=ALPHA)
corr_global_df.insert(0, 'Scope', 'All')
corr_global_df.insert(1, 'Thyroid Status', 'All')
corr_group_dfs = []
for lvl in status_levels:
    sub = df_work[df_work[status_col] == lvl].copy()
    corr_sub = winsorized_pearson_selected_pairs(sub, pairs=target_pairs, winsor_p=WINSOR_P)
    corr_sub = add_fdr_adjustments(corr_sub, alpha=ALPHA)
    corr_sub.insert(0, 'Scope', 'Group')
    corr_sub.insert(1, 'Thyroid Status', str(lvl))
    corr_group_dfs.append(corr_sub)
corr_groups_df = pd.concat(corr_group_dfs, ignore_index=True) if corr_group_dfs else pd.DataFrame()
corr_all_df = pd.concat([corr_global_df, corr_groups_df], ignore_index=True)
comparison_summary = corr_all_df.groupby('Thyroid Status', observed=True, sort=False).agg(Tests=('p_value_raw', 'count'), Unadjusted_p_lt_005=('significant_raw', 'sum'), BH_5_percent=('significant_bh', 'sum')).reset_index()


def corr_df_to_rect_matrices(corr_df, row_variables, column_variables, p_col, alpha):
    r_mat = pd.DataFrame(np.nan, index=row_variables, columns=column_variables, dtype=float)
    p_mat = pd.DataFrame(np.nan, index=row_variables, columns=column_variables, dtype=float)
    for _, row in corr_df.iterrows():
        x, y = (row['Variable_X'], row['Variable_Y'])
        if x in row_variables and y in column_variables:
            r_mat.loc[x, y] = row['r']
            p_mat.loc[x, y] = row[p_col]
        if y in row_variables and x in column_variables:
            r_mat.loc[y, x] = row['r']
            p_mat.loc[y, x] = row[p_col]
    r_sig = r_mat.mask((p_mat >= alpha) | p_mat.isna())
    return (r_mat, p_mat, r_sig)
r_all, p_all_bh, r_all_sig = corr_df_to_rect_matrices(corr_global_df, row_variables=circulating_or_cols, column_variables=rna_target_cols, p_col='p_value_bh', alpha=ALPHA)
p_all_raw = corr_df_to_rect_matrices(corr_global_df, row_variables=circulating_or_cols, column_variables=rna_target_cols, p_col='p_value_raw', alpha=ALPHA)[1]
group_mats = []
for lvl in status_levels:
    df_g = corr_groups_df[corr_groups_df['Thyroid Status'] == str(lvl)].copy()
    r_g, p_g_bh, r_g_sig = corr_df_to_rect_matrices(df_g, row_variables=circulating_or_cols, column_variables=rna_target_cols, p_col='p_value_bh', alpha=ALPHA)
    p_g_raw = corr_df_to_rect_matrices(df_g, row_variables=circulating_or_cols, column_variables=rna_target_cols, p_col='p_value_raw', alpha=ALPHA)[1]
    group_mats.append((str(lvl), r_g, p_g_bh, r_g_sig, p_g_raw))

# %% Correlation figures and exports
OUT_XLSX='Figure_4_Directed_Correlation_Statistics.xlsx'
plot_blocks=[('All samples',r_all,p_all_bh,r_all_sig)]+[(g,r,p,rs) for g,r,p,rs,_ in group_mats]
def correlation_plot(name):
    fig,axes=plt.subplots(2,2,figsize=(12.6,8.3),layout='constrained')
    fig.get_layout_engine().set(w_pad=.10,h_pad=.12,wspace=.16)
    cmap=plt.get_cmap('coolwarm');normcolor=matplotlib.colors.Normalize(-1,1)
    blocks=plot_blocks
    for index,(label,r,p,_) in enumerate(blocks):
        ax=axes.flat[index];a=r.to_numpy();b=p.to_numpy();nr,nc=a.shape
        for i in range(nr):
            for j in range(nc):
                if not np.isfinite(a[i,j]):
                    if r.index[i]!=r.columns[j]:
                        ax.text(j,i,'×',ha='center',va='center',fontsize=8,color='.25')
                    continue
                sig=np.isfinite(b[i,j]) and b[i,j]<ALPHA
                ax.scatter(j,i,s=330*abs(a[i,j]),color=cmap(normcolor(a[i,j])),
                           edgecolor='black' if sig else 'none',linewidth=.8,
                           alpha=1 if sig else .25)
                if sig:
                    ax.annotate(f'{a[i,j]:.2f}',(j,i),xytext=(0,12),textcoords='offset points',
                                ha='center',va='bottom',fontsize=7)
        ax.set(xlim=(-.5,nc-.5),ylim=(nr-.5,-.5));ax.set_xticks(range(nc),[format_plot_label(x) for x in r.columns],rotation=90,fontsize=7);ax.set_yticks(range(nr),[format_plot_label(x) for x in r.index],fontsize=8)
        panel_letter = 'ABCD'[index]
        ax.set_title(panel_letter, loc='left', fontweight='bold')
        ax.set_title(label, loc='center')
        ax.set_xticks(np.arange(nc - 1) + .5, minor=True)
        ax.set_yticks(np.arange(nr - 1) + .5, minor=True)
        ax.tick_params(which='minor', length=0)
        ax.set_axisbelow(True)
        ax.grid(which='minor', color='.88', linewidth=.6)
        ax.spines[['top','right']].set_visible(False)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=normcolor,cmap=cmap),ax=list(axes.flat),shrink=.6,label='Correlation r')
    savefig(fig,name)
correlation_plot('Figure_4_Directed_Correlation_Bubbles.png')
with pd.ExcelWriter(OUT_XLSX,engine='openpyxl') as writer:
    for name,table in {'comparison_summary':comparison_summary,'correlations':corr_all_df}.items():export_values(table).to_excel(writer, sheet_name=name, index=False)
    matrices=[('all',r_all,p_all_raw,p_all_bh)]+[(g,r,raw,p) for g,r,p,_,raw in group_mats]
    for label,r,raw,p in matrices:
        for prefix,matrix in [('r',r),('p_raw',raw),('p_BH',p)]:export_values(matrix).to_excel(writer, sheet_name=sheet(prefix + '_' + label))
    finish_excel(writer)

# %% Supplement: measured TSH sensitivity analysis and Table S1
globals().update(sensitivity())

# %% Export study data and supplementary analyses
# Recompute the original sensitivity task when its results are not in memory.
if not all(name in globals() for name in ['sens_sources','sens_measured','sens_table_s1','sens_observations','sens_descriptives','sens_tsh_results','sens_correlations','sens_review','sens_notes','sens_group_test','sens_versions']):globals().update(sensitivity())
OUT_SUPPLEMENT_XLSX='Supplementary_Study_Data_and_Analyses.xlsx'
supp_study=df_raw.copy();supp_study.insert(1,'Thyroid_status_label',sens_sources['Group'])
supp_study['TSH_measured_available']=sens_measured;supp_study['Manuscript_TSH_source']=np.where(sens_measured,'Measured','ANN-filled')
supp_dictionary=pd.DataFrame([dict(Field=c,Analysis_label=FIELD_TO_LABEL.get(c,c),SPSS_label=(meta.column_names_to_labels or {}).get(c,''),Value_labels=str((meta.variable_value_labels or {}).get(c,{})),Nonmissing_n=int(df_raw[c].notna().sum())) for c in df_raw])
supp_dictionary.insert(0,'Source column number',range(1,len(supp_dictionary)+1))
supp_tables = {'Study_data': supp_study, 'Table_S1': sens_table_s1, 'TSH_observations': sens_observations.assign(**{'TSH source explanation': 'Measured TSH was retained in the manuscript (imputed) dataset; ANN predictions filled only missing measurements. Thus measured and manuscript TSH coincide in these animals (manuscript p. 3, lines 123-126).'}), 'TSH_descriptives': sens_descriptives, 'TSH_correlations': sens_tsh_results, 'Data_dictionary': supp_dictionary}
labels=dict(FIELD_TO_LABEL)
labels.update({'Measured':'Experimentally measured TSH (ng/mL)','ANN':'ANN-predicted TSH (ng/mL)','Manuscript':'Manuscript TSH (ng/mL)','ANN_minus_measured':'ANN minus measured TSH (ng/mL)','1ª corrida':'Run 1','2ª corrida':'Run 2','3ª corrida':'Run 3','1º corrida':'Run 1','2º corrida':'Run 2','3º corrida':'Run 3','Status':'Thyroid status code','Thyroid_status_label':'Thyroid status','TSH_measured_available':'Measured TSH available','Manuscript_TSH_source':'Manuscript TSH source','Predict TSH (ng/mL)':'ANN-predicted TSH (ng/mL)','Quantified TSH (ng/mL)':'Experimentally measured TSH (ng/mL)','Imputed TSH (ng/mL)':'Manuscript TSH (ng/mL)'})
def english(text):
    if not isinstance(text,str):return text
    seen=set()
    while text in labels and text not in seen:seen.add(text);text=labels[text]
    return text
for name,table in list(supp_tables.items()):
    table=table.copy()
    for c in table.select_dtypes(include=['object','string']):table[c]=table[c].map(english)
    supp_tables[name]=table.rename(columns=english)
excel(OUT_SUPPLEMENT_XLSX,supp_tables)

# %% ANN_Nonlinear_RegressionModel
def measured_tsh_regression_results(df_raw):
    def predict(train_x, train_y, test_x, log_response):
        if len(train_y) < train_x.shape[1]:
            return np.full(len(test_x), np.nan)
        target = np.log(train_y) if log_response else train_y
        coefficients, _, rank, _ = np.linalg.lstsq(train_x, target, rcond=None)
        if rank < train_x.shape[1]:
            return np.full(len(test_x), np.nan)
        fitted = test_x @ coefficients
        with np.errstate(over='ignore'):
            return np.exp(fitted) if log_response else fitted

    def metrics(observed, predicted):
        if not len(observed) or not np.isfinite(predicted).all():
            return np.nan, np.nan, np.nan, np.nan
        error = observed - predicted
        sse = error @ error
        centered = observed - observed.mean()
        sst = centered @ centered
        return (1 - sse / sst if sst > 0 else np.nan, np.abs(error).mean(),
                np.sqrt(sse / len(error)), np.abs(error).max())

    def evaluate(name, features, log_response):
        design = np.column_stack((np.ones(len(measured)), *features))
        eligible = np.isfinite(measured) & np.isfinite(design).all(axis=1)
        if log_response:
            eligible &= measured > 0
        x, y = design[eligible], measured[eligible]
        fitted = predict(x, y, x, log_response)
        held_out = np.empty(len(y))
        for index in range(len(y)):
            train = np.arange(len(y)) != index
            held_out[index] = predict(x[train], y[train], x[index:index+1], log_response)[0]
        return (name, design.shape[1], *metrics(y, fitted)[:3], *metrics(y, held_out))

    measured, t4, t3 = df_raw[['qTSH_ngml', 'T4', 'T3']].apply(
        pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float).T
    with np.errstate(divide='ignore', invalid='ignore'):
        log_t4, log_t3 = np.log(t4), np.log(t3)
        inv_t4, inv_t3 = 1 / t4, 1 / t3

    # Compound, Growth and Exponential are equivalent under log-response least squares.
    rows = []
    for hormone, x, log_x, inv_x in [('T4', t4, log_t4, inv_t4), ('T3', t3, log_t3, inv_t3)]:
        models = [
            ('Linear', (x,), False),
            ('Logarithmic', (log_x,), False),
            ('Inverse', (inv_x,), False),
            ('Quadratic', (x, x**2), False),
            ('Cubic', (x, x**2, x**3), False),
            ('Compound', (x,), True),
            ('Power', (log_x,), True),
            ('S', (inv_x,), True),
            ('Growth', (x,), True),
            ('Exponential', (x,), True),
        ]
        rows.extend(evaluate(f'{hormone} {name}', features, log_response)
                    for name, features, log_response in models)

    both_models = [
        ('linear', (t4, t3), False),
        ('log', (log_t4, log_t3), False),
        ('inverse', (inv_t4, inv_t3), False),
        ('power', (log_t4, log_t3), True),
        ('exponential', (t4, t3), True),
        ('S', (inv_t4, inv_t3), True),
        ('linear+interaction', (t4, t3, t4*t3), False),
        ('quadratic additive', (t4, t3, t4**2, t3**2), False),
        ('quadratic full', (t4, t3, t4**2, t3**2, t4*t3), False),
        ('log+interaction', (log_t4, log_t3, log_t4*log_t3), False),
        ('reciprocal+interaction', (inv_t4, inv_t3, inv_t4*inv_t3), False),
    ]
    rows.extend(evaluate(f'Both {name}', features, log_response)
                for name, features, log_response in both_models)
    return pd.DataFrame(rows, columns=[
        'Regression Models', 'Parametros', 'R2_ajuste_original', 'MAE_ajuste',
        'RMSE_ajuste', 'R2_LOOCV', 'MAE_LOOCV', 'RMSE_LOOCV', 'Max_erro_LOOCV',
    ])


ann_destination = results_path / 'Supplementary_Study_Data_and_Analyses.xlsx'
if not ann_destination.is_file():
    raise FileNotFoundError('Run Export study data and supplementary analyses first.')
ann_regression_results = measured_tsh_regression_results(df_raw)
with pd.ExcelWriter(ann_destination, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    export_values(ann_regression_results).to_excel(
        writer, sheet_name='ANN_Nonlinear_RegressionModel', index=False)
    finish_excel(writer)

# %% Supplement: 5000 directed paired bootstrap resamples and 95% intervals
import hashlib
import shutil
from openpyxl import load_workbook
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 261027
BOOTSTRAP_MIN_VALID_FRACTION = 0.95
bootstrap_required = ['df_work','status_col','corr_all_df']
analysis_namespace = globals().copy()
bootstrap_missing = [key for key in bootstrap_required if key not in analysis_namespace]
if bootstrap_missing:
    raise RuntimeError('Run the preceding analysis cells first: ' + ', '.join(bootstrap_missing))
bootstrap_destination = Path(OUT_SUPPLEMENT_XLSX).resolve()
if not bootstrap_destination.is_file():
    raise FileNotFoundError('Create the supplementary workbook in the preceding export section first.')

def bootstrap_coefficient(x, y, method):
    if method == 'Winsorized Pearson (20%)':
        low, high = np.quantile(x, [WINSOR_P, 1 - WINSOR_P], axis=-1, keepdims=True)
        x = np.clip(x, low, high)
        low, high = np.quantile(y, [WINSOR_P, 1 - WINSOR_P], axis=-1, keepdims=True)
        y = np.clip(y, low, high)
    elif method == 'Spearman (exploratory)':
        x = stats.rankdata(x, axis=-1)
        y = stats.rankdata(y, axis=-1)
    else:
        raise ValueError(f'Unsupported bootstrap method: {method}')
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    denominator = np.sqrt((xc * xc).sum(axis=-1) * (yc * yc).sum(axis=-1))
    return np.divide((xc * yc).sum(axis=-1), denominator, out=np.full(np.shape(denominator), np.nan), where=denominator > 0)
bootstrap_cache = {}

def bootstrap_pair_interval(cohort, row):
    names = sorted([row['Variable_X'], row['Variable_Y']])
    pair = cohort[names].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(pair)
    if n != int(row['n']):
        raise ValueError(f"Pair count changed for {names}: {n} versus {row['n']}")
    values = pair.to_numpy(dtype=float)
    method = row['Method']
    signature = hashlib.sha256(method.encode('utf-8') + str(BOOTSTRAP_SEED).encode('ascii') + values.astype('<f8').tobytes()).digest()
    if signature in bootstrap_cache:
        return bootstrap_cache[signature].copy()
    result = {'CI95_low': np.nan, 'CI95_high': np.nan, 'Resamples_requested': BOOTSTRAP_RESAMPLES, 'Valid_resamples': 0, 'Degenerate_resamples': 0, 'CI_status': 'Not estimable: fewer than 3 complete pairs', 'CI_method': 'Paired percentile 95%; per-resample winsorization; CI requires 95% valid resamples'}
    if n >= 3:
        x, y = values.T
        observed = float(bootstrap_coefficient(x, y, method))
        expected = float(row['r'])
        if np.isfinite(observed) != np.isfinite(expected) or (np.isfinite(observed) and (not np.isclose(observed, expected, atol=1e-10))):
            raise ValueError(f'Coefficient changed for {names}: {observed} versus {expected}')
        result['CI_status'] = 'Not estimable: constant original transformed pair'
        if np.isfinite(observed):
            rng = np.random.default_rng(int.from_bytes(signature[:8], 'little'))
            indices = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
            estimates = bootstrap_coefficient(x[indices], y[indices], method)
            valid = np.isfinite(estimates)
            result['Valid_resamples'] = int(valid.sum())
            result['Degenerate_resamples'] = int((~valid).sum())
            result['CI_status'] = 'Unavailable: more than 5% degenerate resamples'
            if valid.mean() >= BOOTSTRAP_MIN_VALID_FRACTION:
                low, high = np.quantile(estimates[valid], [0.025, 0.975])
                result['CI95_low'], result['CI95_high'] = (float(low), float(high))
                result['CI_status'] = 'Exploratory: fewer than 6 pairs; very unstable interval' if n < 6 else 'Exploratory percentile interval; small samples may be unstable'
    bootstrap_cache[signature] = result.copy()
    return result
def bootstrap_results_table(results):
    rows=[]
    for _,row in results.iterrows():
        scenario=row['Thyroid Status']
        cohort=df_work if row['Scope']=='All' else df_work.loc[df_work[status_col].astype(str)==str(scenario)]
        interval=bootstrap_pair_interval(cohort,row)
        rows.append(dict(Scenario=str(scenario),Method=row['Method'],Variable_X=row['Variable_X'],Variable_Y=row['Variable_Y'],n=int(row['n']),Correlation=row['r'],**interval,p_value_raw=row['p_value_raw'],p_value_bh=row['p_value_bh'],BH_family_size=int(row['BH_family_size']),Significant_BH_005=bool(row['p_value_bh']<ALPHA),Bootstrap_seed=BOOTSTRAP_SEED))
    table=pd.DataFrame(rows)
    table[['Variable_X','Variable_Y']]=table[['Variable_X','Variable_Y']].replace({'Imputed TSH (ng/mL)':'Manuscript TSH (ng/mL)'})
    leading=['Scenario','Method','Variable_X','Variable_Y','n','Correlation','CI95_low','CI95_high','p_value_raw','p_value_bh','BH_family_size','Significant_BH_005']
    return table[leading+[c for c in table if c not in leading]]
bootstrap_table=bootstrap_results_table(corr_all_df)
with pd.ExcelWriter(OUT_SUPPLEMENT_XLSX,engine='openpyxl',mode='a',if_sheet_exists='replace') as writer:
    allowed={'Study_data','Table_S1','TSH_observations','TSH_descriptives','TSH_correlations','Data_dictionary','Directed_bootstrap','ANN_Nonlinear_RegressionModel'}
    for unwanted in set(writer.book.sheetnames)-allowed:del writer.book[unwanted]
    export_values(bootstrap_table).to_excel(writer,sheet_name='Directed_bootstrap',index=False)
    finish_excel(writer)
