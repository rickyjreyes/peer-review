#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

SYSTEMS = ("peer_review", "open_triage")
SCENARIOS = ("baseline", "finance", "government", "fame", "entropy", "attribution", "all")
THETA = {
    "detection_intercept": -0.9696890045,
    "reviewer_ability_sd": 0.9640201166,
    "error_difficulty_sd": 0.4858557933,
    "merit_signal": 1.6465977861,
    "rating_noise_sd": 1.8786920115,
    "recommendation_threshold": -1.4033745419,
    "positive_outcome_bias": 2.5808991219,
    "detected_error_penalty": 6.3086866766,
    "trial_base_merit": 2.5399526232,
}


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-30,30)))

def logit(p):
    p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))

def softmax(x):
    z=x-np.max(x); w=np.exp(np.clip(z,-50,50)); return w/w.sum()

def gini(x):
    x=np.asarray(x,float)
    if len(x)==0 or np.all(x==0): return 0.0
    x=np.sort(np.maximum(x,0)); n=len(x)
    return float((2*np.sum((np.arange(1,n+1))*x)/(n*x.sum()))-(n+1)/n)


def params_for_world(rng):
    return {
        "false_share": rng.uniform(.20,.50),
        "mixed_share": rng.uniform(.10,.30),
        "fraud_share": rng.uniform(0,.06),
        "reviewers": int(rng.integers(2,4)),
        "clarity_bias": rng.uniform(.10,.80),
        "prestige_bias": rng.uniform(.0,1.0),
        "novelty_penalty": rng.uniform(.0,1.0),
        "career_conformity": rng.uniform(.0,.8),
        "rho": rng.uniform(.08,.60),
        "exploration": rng.uniform(.15,.55),
        "evidence_noise": rng.uniform(.75,1.8),
        "learning": rng.uniform(.40,1.1),
        "recognition": rng.uniform(.68,.84),
        "evals_per_paper": rng.uniform(1.0,2.0),
        "finance_strength": rng.uniform(.25,1.10),
        "government_strength": rng.uniform(.25,1.10),
        "fame_strength": rng.uniform(.20,1.00),
        "entropy_rate": rng.uniform(.025,.14),
        "capture_rate": rng.uniform(.05,.35),
        "adapt_rate": rng.uniform(.08,.28),
        "certification_reward": rng.uniform(.15,.55),
        "topic_count": int(rng.integers(10,19)),
    }


def scenario_flags(name):
    return {
        "finance": name in ("finance","all"),
        "government": name in ("government","all"),
        "fame": name in ("fame","all"),
        "entropy": name in ("entropy","all"),
        "attribution": name in ("attribution","all"),
    }


def make_opportunities(seed, generation, p, pool):
    # Common latent opportunity pool for all institutions/scenarios.
    rng=np.random.default_rng(np.random.SeedSequence([seed,generation,731]))
    mixed=min(p["mixed_share"], .85-p["false_share"])
    truth=rng.choice([0.0,.5,1.0], size=pool, p=[p["false_share"],mixed,1-p["false_share"]-mixed])
    novelty=rng.beta(2,5,pool)
    prestige=rng.beta(2,5,pool)
    independent=(rng.random(pool)<.36) & (prestige<.45)
    clarity=rng.beta(4,2,pool)
    popularity=rng.beta(2,2,pool)
    commercial=rng.beta(2,2,pool)
    government=rng.beta(2,2,pool)
    topic=rng.integers(0,p["topic_count"],pool)
    positive=rng.random(pool)<.52
    fraud=(rng.random(pool)<p["fraud_share"]) & (truth<1)
    intrinsic=rng.lognormal(0,1,pool)*(1+4.5*novelty**2)
    # Error prevalence is structural, reviewer detection is calibrated.
    err_prob=np.where(truth>=.999,rng.uniform(.07,.22),np.where(truth<=.001,rng.uniform(.52,.82),rng.uniform(.25,.52)))
    err_prob=np.clip(err_prob+.08*fraud, .01,.98)
    major_errors=rng.random((pool,9))<err_prob[:,None]
    error_difficulty=rng.normal(size=(pool,9))
    return dict(truth=truth,novelty=novelty,prestige=prestige,independent=independent,clarity=clarity,
                popularity=popularity,commercial=commercial,government=government,topic=topic,positive=positive,
                fraud=fraud,intrinsic=intrinsic,major_errors=major_errors,error_difficulty=error_difficulty)


def formal_review(rng, papers, p, idx):
    n=len(idx); scores=[]
    truth=papers["truth"][idx]; clarity=papers["clarity"][idx]; prestige=papers["prestige"][idx]; novelty=papers["novelty"][idx]
    positive=papers["positive"][idx].astype(float)
    errors=papers["major_errors"][idx]; diff=papers["error_difficulty"][idx]
    for _ in range(p["reviewers"]):
        ability=rng.normal(0,THETA["reviewer_ability_sd"],n)
        logits=THETA["detection_intercept"]+ability[:,None]-THETA["error_difficulty_sd"]*diff
        detected=errors & (rng.random(errors.shape)<sigmoid(logits))
        detected_fraction=detected.mean(1)
        institutional=(p["clarity_bias"]*(clarity-.5)+p["prestige_bias"]*(prestige-.5)-p["novelty_penalty"]*novelty+p["career_conformity"]*(prestige-novelty))
        score=(THETA["trial_base_merit"]+THETA["merit_signal"]*((truth-.5)*2)+THETA["positive_outcome_bias"]*positive-
               THETA["detected_error_penalty"]*detected_fraction+institutional+rng.normal(0,THETA["rating_noise_sd"],n))
        scores.append(score)
    mean=np.mean(scores,axis=0)
    return mean>THETA["recommendation_threshold"],mean


def choose_projects(rng, papers, p, flags, culture, k):
    # Researchers do not observe truth. They choose opportunities by visible/reward-linked attributes.
    appeal=(.35*papers["clarity"]+.25*papers["novelty"]+.15*papers["popularity"]+.10*(1-papers["prestige"])+rng.normal(0,.20,len(papers["truth"])))
    if flags["attribution"]:
        appeal -= culture.get("independent_exit",0.0)*papers["independent"].astype(float)
    if flags["finance"]:
        appeal += p["finance_strength"]*(1+culture["finance"])*papers["commercial"]
    if flags["government"]:
        appeal += p["government_strength"]*(1+culture["government"])*papers["government"]
    if flags["fame"]:
        # Fame rewards prestige, popular topics, and conformity; peer-review culture can amplify certification-linked prestige.
        appeal += p["fame_strength"]*(1+culture["fame"])*(.55*papers["prestige"]+.35*papers["popularity"]-.30*papers["novelty"])
    # Gumbel top-k gives stochastic weighted choice without replacement.
    g=-np.log(-np.log(np.clip(rng.random(len(appeal)),1e-12,1-1e-12)))
    return np.argpartition(appeal+g, -k)[-k:]


def evaluate_generation(rng, papers, idx, p, system, flags, culture, backlog):
    n=len(idx); accepted, _=formal_review(rng,papers,p,idx)
    if system=="open_triage": accepted=np.ones(n,dtype=bool)
    confidence=np.full(n,.5); counts_total=np.zeros(n,int)
    # Equal lifetime expert-action budget. Peer review consumes gate actions.
    total_actions=max(n*p["reviewers"]+1, int(round(p["evals_per_paper"]*n*7)))
    gate_actions=n*p["reviewers"] if system=="peer_review" else 0
    downstream=max(n, total_actions-gate_actions)
    rounds=7
    per_round=max(1,downstream//rounds)
    for t in range(rounds):
        prestige=papers["prestige"][idx]; novelty=papers["novelty"][idx]; clarity=papers["clarity"][idx]; popularity=papers["popularity"][idx]
        if system=="peer_review":
            appeal=1.05*confidence+.75*prestige+.20*clarity+.15*popularity
            w=softmax(2.2*appeal)
            w[~accepted]*=p["rho"]
            w=w/w.sum()
        else:
            merit=.30*clarity+.28*novelty+.25*confidence+.12*(1-prestige)+.05*popularity
            w=softmax(1.8*merit)
            w=(1-p["exploration"])*w+p["exploration"]*(np.ones(n)/n)
        c=rng.multinomial(per_round,w); counts_total+=c
        seen=c>0
        if np.any(seen):
            truth=papers["truth"][idx][seen]
            signal=(truth-.5)*2+rng.normal(0,p["evidence_noise"]/np.sqrt(c[seen]),seen.sum())
            repl=rng.random(seen.sum())<(1-np.exp(-.28*c[seen])); signal+=repl*(truth-.5)*.75
            fraud=papers["fraud"][idx][seen]; detected=fraud & (rng.random(seen.sum())<(1-np.exp(-.18*c[seen])))
            signal += fraud*.35-detected*1.5
            confidence[seen]=sigmoid(logit(confidence[seen])+p["learning"]*signal)
        confidence[~seen]=.997*confidence[~seen]+.003*.5
    recognized=confidence>=p["recognition"]
    truth=papers["truth"][idx]; intrinsic=papers["intrinsic"][idx]
    true_value=(truth*intrinsic)
    recovered=true_value[recognized].sum()
    total=true_value.sum()+1e-12
    false_rec=((recognized)&(truth==0)).sum()/max(1,(truth==0).sum())

    # Payoffs drive next-generation project choice. No truth is used here except through noisy recognition.
    reward=recognized.astype(float)+.10*np.log1p(counts_total)
    if flags["finance"]: reward += .45*papers["commercial"][idx]*recognized
    if flags["government"]: reward += .45*papers["government"][idx]*recognized
    if flags["fame"]:
        reward += .35*papers["prestige"][idx]*recognized
        if system=="peer_review": reward += p["certification_reward"]*accepted
    culture2=culture.copy()
    ar=p["adapt_rate"]
    if flags["finance"]:
        culture2["finance"]=(1-ar)*culture["finance"]+ar*np.clip(np.corrcoef(reward,papers["commercial"][idx])[0,1] if np.std(reward)>0 else 0,-.5,.8)
    if flags["government"]:
        culture2["government"]=(1-ar)*culture["government"]+ar*np.clip(np.corrcoef(reward,papers["government"][idx])[0,1] if np.std(reward)>0 else 0,-.5,.8)
    if flags["fame"]:
        fam=.6*papers["prestige"][idx]+.4*papers["popularity"][idx]-.25*papers["novelty"][idx]
        culture2["fame"]=(1-ar)*culture["fame"]+ar*np.clip(np.corrcoef(reward,fam)[0,1] if np.std(reward)>0 else 0,-.5,.8)

    # Persistent unrecognized true ideas. Entropy lowers rediscoverability; rejected peer-review work decays faster.
    new_backlog=[]
    for j,orig in enumerate(idx):
        if truth[j]>0 and not recognized[j]:
            vis=1.0
            if flags["entropy"]:
                vis*=math.exp(-p["entropy_rate"]*(1.0 + (0.65 if system=="peer_review" and not accepted[j] else 0.0)))
            new_backlog.append({"topic":int(papers["topic"][orig]),"value":float(true_value[j]),"independent":bool(papers["independent"][orig]),"prestige":float(papers["prestige"][orig]),"visibility":vis,"age":1,"credited":False})
    surviving=[]
    attribution_credit=0.0; captured_value=0.0; rediscovered_value=0.0
    recognized_topics=papers["topic"][idx][recognized]
    recognized_prestige=papers["prestige"][idx][recognized]
    for item in backlog:
        if flags["entropy"]:
            extra=1.0 + (0.45 if system=="peer_review" else 0.0)
            item["visibility"]*=math.exp(-p["entropy_rate"]*extra)
        item["age"]+=1
        matches=np.flatnonzero(recognized_topics==item["topic"])
        if len(matches):
            maxp=float(np.max(recognized_prestige[matches]))
            chance=min(.85,.08+.35*item["visibility"])
            if rng.random()<chance:
                rediscovered_value+=item["value"]
                if item["independent"]:
                    if flags["attribution"]:
                        cap=p["capture_rate"]*sigmoid(np.array([(maxp-item["prestige"])*4 + .30*item["age"] - 1.0]))[0]
                        if rng.random()<cap:
                            captured_value+=item["value"]
                        else: attribution_credit+=item["value"]
                    else: attribution_credit+=item["value"]
                continue
        if item["visibility"]>.02 and item["age"]<12: surviving.append(item)
    surviving.extend(new_backlog)

    independent_true=(truth>0)&papers["independent"][idx]
    independent_value=true_value[independent_true].sum()+1e-12
    independent_direct=true_value[independent_true & recognized].sum()
    if flags["attribution"]:
        denom=max(1e-12, independent_direct+attribution_credit+captured_value)
        capture_pressure=captured_value/denom
        culture2["independent_exit"]=(1-ar)*culture.get("independent_exit",0.0)+ar*np.clip(2.0*capture_pressure,0,1.5)
    attribution_integrity=(independent_direct+attribution_credit)/(independent_value+sum(x["value"] for x in backlog if x["independent"])+1e-12)
    return {
        "recovered": recovered, "available": total, "false_rec": false_rec,
        "recognized": int(recognized.sum()), "attention_gini": gini(counts_total),
        "novelty_pursued": float(papers["novelty"][idx].mean()),
        "independent_value_direct": float(independent_direct),
        "attribution_credit": float(attribution_credit), "captured_value": float(captured_value),
        "rediscovered_value": float(rediscovered_value), "attribution_integrity": float(attribution_integrity),
        "accepted_share": float(accepted.mean()) if system=="peer_review" else 1.0,
        "culture": culture2, "backlog": surviving,
    }


def simulate_system(world_seed,p,scenario,system,generations,pool,pursued):
    flags=scenario_flags(scenario); culture={"finance":0.0,"government":0.0,"fame":0.0,"independent_exit":0.0}; backlog=[]
    sums={k:0.0 for k in ["recovered","available","false_rec","captured","rediscovered","independent_credit"]}
    nov=[]; gin=[]; attr=[]; accepts=[]
    for g in range(generations):
        papers=make_opportunities(world_seed,g,p,pool)
        rng_choice=np.random.default_rng(np.random.SeedSequence([world_seed,g,1001]))
        idx=choose_projects(rng_choice,papers,p,flags,culture,pursued)
        rng_eval=np.random.default_rng(np.random.SeedSequence([world_seed,g,2003]))
        out=evaluate_generation(rng_eval,papers,idx,p,system,flags,culture,backlog)
        culture=out["culture"]; backlog=out["backlog"]
        sums["recovered"]+=out["recovered"]+out["rediscovered_value"]
        sums["available"]+=out["available"]
        sums["false_rec"]+=out["false_rec"]
        sums["captured"]+=out["captured_value"]
        sums["rediscovered"]+=out["rediscovered_value"]
        sums["independent_credit"]+=out["independent_value_direct"]+out["attribution_credit"]
        nov.append(out["novelty_pursued"]); gin.append(out["attention_gini"]); attr.append(out["attribution_integrity"]); accepts.append(out["accepted_share"])
    return {
        "truth_value_recovery": sums["recovered"]/(sums["available"]+1e-12),
        "recovered_true_value": sums["recovered"],
        "available_true_value": sums["available"],
        "missed_true_value": max(0.0, sums["available"]-sums["recovered"]),
        "false_recognition_rate": sums["false_rec"]/generations,
        "captured_value_share": sums["captured"]/(sums["available"]+1e-12),
        "rediscovered_value_share": sums["rediscovered"]/(sums["available"]+1e-12),
        "independent_credit_share": sums["independent_credit"]/(sums["available"]+1e-12),
        "mean_novelty_pursued": float(np.mean(nov)), "attention_gini": float(np.mean(gin)),
        "attribution_integrity": float(np.mean(attr)), "accepted_share": float(np.mean(accepts)),
        "culture_finance": culture["finance"], "culture_government": culture["government"], "culture_fame": culture["fame"], "culture_independent_exit": culture.get("independent_exit",0.0),
        "backlog_value": float(sum(x["value"] for x in backlog)),
    }


def run(seed,worlds,generations,pool,pursued):
    rows=[]
    for w in range(worlds):
        prng=np.random.default_rng(np.random.SeedSequence([seed,w,19])); p=params_for_world(prng)
        world_seed=int(np.random.SeedSequence([seed,w]).generate_state(1)[0])
        for sc in SCENARIOS:
            for sys in SYSTEMS:
                out=simulate_system(world_seed,p,sc,sys,generations,pool,pursued)
                rows.append({"world":w,"scenario":sc,"system":sys,**out})
    df=pd.DataFrame(rows)
    # Paired differences open_triage - peer_review; positive recovery difference favors open triage.
    metrics=["truth_value_recovery","false_recognition_rate","captured_value_share","rediscovered_value_share","independent_credit_share","mean_novelty_pursued","attention_gini","attribution_integrity","backlog_value"]
    diffs=[]
    for sc in SCENARIOS:
        sub=df[df.scenario==sc].set_index(["world","system"])
        for w in range(worlds):
            a=sub.loc[(w,"open_triage")]; b=sub.loc[(w,"peer_review")]
            row={"world":w,"scenario":sc}
            for m in metrics: row[m+"_diff_open_minus_peer"]=float(a[m]-b[m])
            diffs.append(row)
    d=pd.DataFrame(diffs)
    summary=[]
    for sc in SCENARIOS:
        ds=d[d.scenario==sc]
        rec=ds["truth_value_recovery_diff_open_minus_peer"]
        summary.append({
            "scenario":sc,
            "mean_recovery_diff":rec.mean(),
            "median_recovery_diff":rec.median(),
            "open_higher_recovery_share":float((rec>0).mean()),
            "peer_higher_recovery_share":float((rec<0).mean()),
            "mean_false_recognition_diff":ds["false_recognition_rate_diff_open_minus_peer"].mean(),
            "mean_attribution_integrity_diff":ds["attribution_integrity_diff_open_minus_peer"].mean(),
            "mean_captured_value_diff":ds["captured_value_share_diff_open_minus_peer"].mean(),
            "mean_novelty_diff":ds["mean_novelty_pursued_diff_open_minus_peer"].mean(),
            "mean_attention_gini_diff":ds["attention_gini_diff_open_minus_peer"].mean(),
        })
    return df,d,pd.DataFrame(summary)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=20260806); ap.add_argument("--worlds",type=int,default=300); ap.add_argument("--generations",type=int,default=10); ap.add_argument("--pool",type=int,default=300); ap.add_argument("--pursued",type=int,default=120); ap.add_argument("--output-dir",type=Path,default=Path("results/scale_incentives")); args=ap.parse_args()
    df,d,s=run(args.seed,args.worlds,args.generations,args.pool,args.pursued); args.output_dir.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.output_dir/"system_results.csv",index=False); d.to_csv(args.output_dir/"paired_differences.csv",index=False); s.to_csv(args.output_dir/"scenario_summary.csv",index=False)
    meta={"seed":args.seed,"worlds":args.worlds,"generations":args.generations,"opportunity_pool_per_generation":args.pool,"projects_pursued_per_generation":args.pursued,"scenarios":list(SCENARIOS),"systems":list(SYSTEMS),"reviewer_calibration":THETA,"interpretive_boundary":"Finance, government, fame, entropy, and attribution-capture strengths are structural sweeps, not empirical estimates. Reviewer behavior uses the committed calibrated parameter set. Equal total expert-action budgets are imposed within each generation."}
    (args.output_dir/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(s.to_string(index=False,float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__": main()
