from __future__ import annotations

import math
import os
import json
import traceback
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:
    ExponentialSmoothing = None


@dataclass
class JobResult:
    name: str
    status: str
    duration_s: float
    summary: str
    data: Any = None
    error: Optional[str] = None


class SPDSEngine:
    """Deterministic analytical core. No LLM/API key required."""

    def __init__(self, df: pd.DataFrame):
        self.raw = df.copy()
        self.df = df.copy()
        self.profile = self._profile()

    def _profile(self) -> Dict[str, Any]:
        df = self.df
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        categorical = [c for c in df.columns if c not in numeric]
        missing = df.isna().sum().sort_values(ascending=False)
        return {
            "rows": int(len(df)), "columns": int(len(df.columns)),
            "numeric": numeric, "categorical": categorical,
            "missing_total": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum()/1e6, 2),
            "missing": missing.head(20).to_dict(),
        }

    def clean(self) -> JobResult:
        import time
        t=time.perf_counter()
        before=len(self.df)
        self.df=self.df.drop_duplicates().copy()
        for c in self.df.select_dtypes(include=np.number):
            if self.df[c].isna().any(): self.df[c]=self.df[c].fillna(self.df[c].median())
        for c in self.df.select_dtypes(exclude=np.number):
            if self.df[c].isna().any():
                mode=self.df[c].mode(dropna=True)
                self.df[c]=self.df[c].fillna(mode.iloc[0] if len(mode) else "Unknown")
        return JobResult("clean", "ok", time.perf_counter()-t,
                         f"Removed {before-len(self.df)} duplicate rows and imputed missing values.",
                         {"rows_after":len(self.df),"rows_removed":before-len(self.df)})

    def eda(self) -> JobResult:
        import time
        t=time.perf_counter(); df=self.df
        nums=df.select_dtypes(include=np.number)
        desc=nums.describe().T if not nums.empty else pd.DataFrame()
        corr=nums.corr(numeric_only=True) if nums.shape[1] >= 2 else pd.DataFrame()
        return JobResult("eda", "ok", time.perf_counter()-t, "Generated statistical profile, distributions and correlations.",
                         {"describe":desc,"correlation":corr})

    def anomalies(self) -> JobResult:
        import time
        t=time.perf_counter(); nums=self.df.select_dtypes(include=np.number)
        if nums.shape[1] < 1 or len(nums) < 20:
            return JobResult("anomalies","skipped",time.perf_counter()-t,"Not enough numeric observations.",{})
        X=nums.replace([np.inf,-np.inf],np.nan).fillna(nums.median()).fillna(0)
        model=IsolationForest(n_estimators=150, contamination="auto", random_state=42, n_jobs=-1)
        labels=model.fit_predict(X)
        scores=-model.score_samples(X)
        out=self.df.copy(); out["_anomaly"]=(labels==-1); out["_anomaly_score"]=scores
        return JobResult("anomalies","ok",time.perf_counter()-t,
                         f"Detected {int((labels==-1).sum())} potentially anomalous rows.",out.sort_values("_anomaly_score",ascending=False).head(100))

    def _candidate_target(self, target: Optional[str]) -> Optional[str]:
        if target and target in self.df.columns: return target
        nums=self.df.select_dtypes(include=np.number).columns
        return nums[-1] if len(nums) else None

    def automl(self, target: Optional[str]=None) -> JobResult:
        import time
        t=time.perf_counter(); target=self._candidate_target(target)
        if not target: return JobResult("automl","skipped",time.perf_counter()-t,"No numeric target available.",{})
        df=self.df.dropna(subset=[target]).copy(); y=df[target]
        X=df.drop(columns=[target]).copy()
        # Convert datetimes to useful calendar features; never pass raw datetime objects into sklearn.
        for c in list(X.select_dtypes(include=["datetime", "datetimetz"]).columns):
            dt=pd.to_datetime(X[c],errors="coerce")
            X[c+"__year"]=dt.dt.year
            X[c+"__month"]=dt.dt.month
            X[c+"__dayofweek"]=dt.dt.dayofweek
            X[c+"__dayofyear"]=dt.dt.dayofyear
            X.drop(columns=[c],inplace=True)
        nums=X.select_dtypes(include=np.number).columns.tolist(); cats=[c for c in X.columns if c not in nums]
        if len(df)<30 or not nums and not cats: return JobResult("automl","skipped",time.perf_counter()-t,"Insufficient usable data.",{})
        transformers=[]
        if nums: transformers.append(("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums))
        if cats: transformers.append(("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cats))
        prep=ColumnTransformer(transformers)
        split=max(1,int(len(df)*0.8)); Xtr,Xte=X.iloc[:split],X.iloc[split:]; ytr,yte=y.iloc[:split],y.iloc[split:]
        # Time-aware split preserves ordering; safer for operational forecasting-like datasets.
        models={"linear":LinearRegression(),"extra_trees":ExtraTreesRegressor(n_estimators=250,random_state=42,n_jobs=-1)}
        rows=[]
        for name,model in models.items():
            pipe=Pipeline([("prep",prep),("model",model)])
            pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
            rows.append({"model":name,"MAE":mean_absolute_error(yte,pred),"RMSE":math.sqrt(mean_squared_error(yte,pred)),"R2":r2_score(yte,pred)})
        res=pd.DataFrame(rows).sort_values("RMSE")
        return JobResult("automl","ok",time.perf_counter()-t,f"Compared {len(rows)} regression models for target '{target}'.",{"target":target,"leaderboard":res})

    def forecast(self, date_col: Optional[str]=None, target: Optional[str]=None, horizon:int=12) -> JobResult:
        import time
        t=time.perf_counter(); df=self.df.copy(); target=self._candidate_target(target)
        if not target: return JobResult("forecast","skipped",time.perf_counter()-t,"No numeric target available.",{})
        if date_col is None:
            candidates=[]
            for c in df.columns:
                parsed=pd.to_datetime(df[c],errors="coerce")
                if parsed.notna().mean()>0.8: candidates.append((c,parsed.notna().mean()))
            date_col=max(candidates,key=lambda x:x[1])[0] if candidates else None
        if not date_col: return JobResult("forecast","skipped",time.perf_counter()-t,"No date column detected.",{})
        s=pd.DataFrame({"date":pd.to_datetime(df[date_col],errors="coerce"),"y":pd.to_numeric(df[target],errors="coerce")}).dropna().sort_values("date")
        s=s.groupby("date",as_index=False)["y"].mean()
        if len(s)<12: return JobResult("forecast","skipped",time.perf_counter()-t,"Need at least 12 dated observations.",{})
        freq=pd.infer_freq(s["date"]) or "D"; idx=pd.date_range(s.date.min(),s.date.max(),freq=freq); y=s.set_index("date")["y"].reindex(idx).interpolate().bfill().ffill()
        h=max(1,min(int(horizon),365))
        if ExponentialSmoothing is not None:
            seasonal=None; sp=None
            if len(y)>=24:
                for p in (12,7,4):
                    if len(y)>=2*p: seasonal="add"; sp=p; break
            try:
                fit=ExponentialSmoothing(y,trend="add",seasonal=seasonal,seasonal_periods=sp,initialization_method="estimated").fit(optimized=True)
                pred=fit.forecast(h)
            except Exception:
                pred=pd.Series(np.repeat(y.tail(min(12,len(y))).mean(),h),index=pd.date_range(y.index[-1],periods=h+1,freq=freq)[1:])
        else:
            pred=pd.Series(np.repeat(y.tail(min(12,len(y))).mean(),h),index=pd.date_range(y.index[-1],periods=h+1,freq=freq)[1:])
        resid=float(np.std(np.diff(y))) if len(y)>2 else float(np.std(y))
        lower=pred-1.96*resid*np.sqrt(np.arange(1,h+1)); upper=pred+1.96*resid*np.sqrt(np.arange(1,h+1))
        out=pd.DataFrame({"date":pred.index,"forecast":pred.values,"lower_95":lower.values,"upper_95":upper.values})
        return JobResult("forecast","ok",time.perf_counter()-t,f"Forecasted '{target}' for {h} periods using {date_col}.",{"target":target,"date_col":date_col,"forecast":out})

    def scenarios(self, target:Optional[str]=None) -> JobResult:
        import time
        t=time.perf_counter(); target=self._candidate_target(target)
        if not target: return JobResult("scenarios","skipped",time.perf_counter()-t,"No numeric target available.",{})
        y=pd.to_numeric(self.df[target],errors="coerce").dropna()
        if len(y)<5: return JobResult("scenarios","skipped",time.perf_counter()-t,"Insufficient target history.",{})
        base=float(y.tail(min(12,len(y))).mean()); trend=float(y.tail(min(12,len(y))).mean()-y.head(min(12,len(y))).mean())/max(1,len(y))
        rows=[]
        for name,mult in [("downside",0.90),("baseline",1.00),("expansion",1.10)]:
            val=max(0,base+trend*len(y))*mult
            rows.append({"scenario":name,"estimated_value":val,"relative_to_baseline":mult-1})
        return JobResult("scenarios","ok",time.perf_counter()-t,"Built downside, baseline and expansion scenarios.",pd.DataFrame(rows))

    def insights(self, eda:JobResult, anomalies:JobResult, automl:JobResult, forecast:JobResult, scenarios:JobResult) -> JobResult:
        import time
        t=time.perf_counter(); bullets=[]
        if eda.data and not eda.data["correlation"].empty:
            c=eda.data["correlation"].abs().where(lambda x:~np.eye(len(x),dtype=bool)).stack().sort_values(ascending=False)
            if len(c):
                a,b=c.index[0]; bullets.append(f"Strongest observed numeric association: {a} ↔ {b} (absolute correlation {c.iloc[0]:.2f}). This is not proof of causation.")
        if anomalies.status=="ok": bullets.append(anomalies.summary)
        if automl.status=="ok":
            lb=automl.data["leaderboard"]; best=lb.iloc[0]; bullets.append(f"Best tested regression model by RMSE: {best['model']} (RMSE {best['RMSE']:.3g}, R² {best['R2']:.3f}).")
        if forecast.status=="ok":
            fc=forecast.data["forecast"]; bullets.append(f"Forecast endpoint: {fc['forecast'].iloc[-1]:.3g}, with a model-based uncertainty band shown in the report.")
        bullets.append("Recommendations should be treated as decision support; validate causal assumptions and operational constraints before acting.")
        return JobResult("insights","ok",time.perf_counter()-t,"Generated evidence-backed findings and caveats.",bullets)

    def run_autonomous(self, target=None, date_col=None, horizon=12, workers=5) -> Dict[str,JobResult]:
        import time
        start=time.perf_counter(); clean=self.clean()
        jobs={}
        funcs={
            "eda":lambda:self.eda(),
            "anomalies":lambda:self.anomalies(),
            "automl":lambda:self.automl(target),
            "forecast":lambda:self.forecast(date_col,target,horizon),
            "scenarios":lambda:self.scenarios(target),
        }
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs={ex.submit(fn):name for name,fn in funcs.items()}
            for fut in as_completed(futs):
                name=futs[fut]
                try: jobs[name]=fut.result()
                except Exception as e: jobs[name]=JobResult(name,"error",0,"Job failed",None,traceback.format_exc())
        jobs["insights"]=self.insights(jobs["eda"],jobs["anomalies"],jobs["automl"],jobs["forecast"],jobs["scenarios"])
        jobs["_meta"]=JobResult("orchestration","ok",time.perf_counter()-start,"Autonomous analysis complete",{"total_seconds":time.perf_counter()-start,"profile":self.profile})
        return jobs


def load_file(path:str)->pd.DataFrame:
    ext=os.path.splitext(path)[1].lower()
    if ext==".csv": return pd.read_csv(path)
    if ext in (".xlsx",".xls"): return pd.read_excel(path)
    if ext==".json": return pd.read_json(path)
    if ext==".parquet": return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {ext}")
