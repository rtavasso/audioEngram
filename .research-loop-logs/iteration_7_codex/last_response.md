**My bets (as of your current evidence)**

1) Predictor baseline shows meaningful predictability (conditional NLL ≪ marginal): **0.55**  
- Biggest uncertainty: multimodality. I think one-step MSE/R² may look bad, but a mixture/flow NLL will likely find *some* conditional structure if it’s there. The Phase 0 near-flatness pulls this down, but the “wrong key/metric” loophole is real.

2) RSSM-style factorization yields `z_dyn` that’s substantially easier for AR than raw Mimi latents: **0.65**  
- This is the part I’m most optimistic about because the KL “innovation budget” + prior-sampled training is a strong inductive bias, and you’re allowed to lower the effective innovation rate by design.

3) If (2) succeeds, Engram-style lookup helps for `z_dyn`: **0.45**  
- Lookup helps when there are lots of **recurrent, compressible local motifs under a stable key**. Even if `z_dyn` becomes AR-friendly, it might be “smooth-but-nonrepeating” (low entropy but not memorizable), in which case memory helps less than you’d hope.

4) Full pipeline beats CALM on quality or efficiency: **0.25**  
- This is where I’d be cautious. CALM’s engineering exists because continuous latent AR is hard; your approach can absolutely reduce hardness, but surpassing a tuned system is a high bar. I’d frame success initially as “simpler training / fewer stabilization tricks / better scaling,” not immediate SOTA quality.

**Most likely failure point:** (1) or (3), depending on what you mean by “meaningful.” I think (2) is the most likely to work in *some* form, but it may not translate into “Engram wins” if the learned state doesn’t have reusable motifs.

---

**Single most belief-updating experiment for the next week**  
Train the **best-effort probabilistic predictor** on current Mimi latents and report:

- `ΔNLL = NLL[p(Δx | full context)] - NLL[p(Δx)]` (teacher-forced), using a mixture-density network or lightweight flow  
- The same `ΔNLL` at multiple horizons (predict `x_{t+k}` or `Δx_{t+k}` for k=1,2,4,8)  
- One free-running rollout metric (even crude), to see teacher-forced vs rollout gap

If `ΔNLL` is ~0 even with a strong predictor, it’s a huge update against the whole “representation is hiding structure at 12.5 Hz” hypothesis. If `ΔNLL` is clearly >0, it strongly supports moving to Option (I) and tells you what horizons/timescales to target.
