phase1 metrics.json for phase3's zdyn:
[
  {
    "horizon_k": 1,
    "slice_name": "all",
    "train": {
      "n": 200000,
      "nll": -190.934343046875,
      "nll_per_dim": -1.491674555053711,
      "nll_baseline": -131.45609314941407,
      "nll_baseline_per_dim": -1.0270007277297974,
      "dnll": -59.47824989746093,
      "dnll_per_dim": -0.4646738273239135,
      "direction_cosine": 0.703725064086914,
      "logmag_mse": 0.14373569545745848,
      "logmag_r2": -0.071724028664099
    },
    "eval": {
      "n": 200000,
      "nll": -192.3472385546875,
      "nll_per_dim": -1.502712801208496,
      "nll_baseline": -131.2340589453125,
      "nll_baseline_per_dim": -1.025266085510254,
      "dnll": -61.11317960937498,
      "dnll_per_dim": -0.47744671569824204,
      "direction_cosine": 0.7084952416801452,
      "logmag_mse": 0.1361570604133606,
      "logmag_r2": -0.09790971093190404
    },
    "rollout": {
      "n_utterances": 16,
      "teacher_forced": {
        "n": 2300,
        "nll": -192.6094178108547,
        "nll_per_dim": -1.5047610766473023,
        "nll_baseline": -132.18902040730353,
        "nll_baseline_per_dim": -1.0327267219320588,
        "dnll": -60.42039740355116,
        "dnll_per_dim": -0.47203435471524346,
        "direction_cosine": 0.7086608846827774,
        "logmag_mse": 0.1369845577033505,
        "logmag_r2": -0.42502354644446827
      },
      "rollout_context": {
        "n": 2300,
        "nll": -123.9101919655178,
        "nll_per_dim": -0.9680483747306078,
        "nll_baseline": -132.18902040730353,
        "nll_baseline_per_dim": -1.0327267219320588,
        "dnll": 8.278828441785734,
        "dnll_per_dim": 0.06467834720145105,
        "direction_cosine": 0.013467972619141422,
        "logmag_mse": 21.77890116459771,
        "logmag_r2": -225.56164676932408
      },
      "gap_nll": 68.6992258453369,
      "gap_dnll": 68.6992258453369
    }
  },
  {
    "horizon_k": 2,
    "slice_name": "all",
    "train": {
      "n": 200000,
      "nll": -161.71146455078124,
      "nll_per_dim": -1.2633708168029785,
      "nll_baseline": -131.45609314941407,
      "nll_baseline_per_dim": -1.0270007277297974,
      "dnll": -30.25537140136717,
      "dnll_per_dim": -0.23637008907318102,
      "direction_cosine": 0.44057073371887207,
      "logmag_mse": 0.8500567122650147,
      "logmag_r2": -5.3382043086941975
    },
    "eval": {
      "n": 200000,
      "nll": -161.605119453125,
      "nll_per_dim": -1.2625399957275392,
      "nll_baseline": -131.2340589453125,
      "nll_baseline_per_dim": -1.025266085510254,
      "dnll": -30.3710605078125,
      "dnll_per_dim": -0.23727391021728517,
      "direction_cosine": 0.44011576730728147,
      "logmag_mse": 0.8350551718521119,
      "logmag_r2": -5.733511868991406
    },
    "rollout": {
      "n_utterances": 16,
      "teacher_forced": {
        "n": 2277,
        "nll": -164.58049709167614,
        "nll_per_dim": -1.2857851335287198,
        "nll_baseline": -134.05348679848706,
        "nll_baseline_per_dim": -1.0472928656131801,
        "dnll": -30.52701029318908,
        "dnll_per_dim": -0.23849226791553968,
        "direction_cosine": 0.4356982531428372,
        "logmag_mse": 0.8343147620370676,
        "logmag_r2": -4.986257893271582
      },
      "rollout_context": {
        "n": 2277,
        "nll": NaN,
        "nll_per_dim": NaN,
        "nll_baseline": -134.05348679848706,
        "nll_baseline_per_dim": -1.0472928656131801,
        "dnll": NaN,
        "dnll_per_dim": NaN,
        "direction_cosine": NaN,
        "logmag_mse": NaN,
        "logmag_r2": NaN
      },
      "gap_nll": NaN,
      "gap_dnll": NaN
    }
  },
  {
    "horizon_k": 4,
    "slice_name": "all",
    "train": {
      "n": 200000,
      "nll": -149.13276833984375,
      "nll_per_dim": -1.1650997526550293,
      "nll_baseline": -131.45609314941407,
      "nll_baseline_per_dim": -1.0270007277297974,
      "dnll": -17.676675190429677,
      "dnll_per_dim": -0.13809902492523185,
      "direction_cosine": 0.1520515224504471,
      "logmag_mse": 5.013652862854004,
      "logmag_r2": -36.38286601251051
    },
    "eval": {
      "n": 200000,
      "nll": -148.67317142578125,
      "nll_per_dim": -1.161509151763916,
      "nll_baseline": -131.2340589453125,
      "nll_baseline_per_dim": -1.025266085510254,
      "dnll": -17.439112480468737,
      "dnll_per_dim": -0.136243066253662,
      "direction_cosine": 0.1522323039674759,
      "logmag_mse": 4.99056680770874,
      "logmag_r2": -39.241701345516155
    },
    "rollout": {
      "n_utterances": 16,
      "teacher_forced": {
        "n": 2380,
        "nll": -150.91435018908075,
        "nll_per_dim": -1.1790183608521934,
        "nll_baseline": -132.98917237570305,
        "nll_baseline_per_dim": -1.03897790918518,
        "dnll": -17.9251778133777,
        "dnll_per_dim": -0.1400404516670133,
        "direction_cosine": 0.15174068968702792,
        "logmag_mse": 4.906245352307912,
        "logmag_r2": -39.13518631460868
      },
      "rollout_context": {
        "n": 2380,
        "nll": NaN,
        "nll_per_dim": NaN,
        "nll_baseline": -132.98917237570305,
        "nll_baseline_per_dim": -1.03897790918518,
        "dnll": NaN,
        "dnll_per_dim": NaN,
        "direction_cosine": NaN,
        "logmag_mse": NaN,
        "logmag_r2": NaN
      },
      "gap_nll": NaN,
      "gap_dnll": NaN
    }
  },
  {
    "horizon_k": 8,
    "slice_name": "all",
    "train": {
      "n": 200000,
      "nll": -144.47997451660157,
      "nll_per_dim": -1.1287498009109498,
      "nll_baseline": -131.91264184570312,
      "nll_baseline_per_dim": -1.0305675144195556,
      "dnll": -12.567332670898452,
      "dnll_per_dim": -0.09818228649139416,
      "direction_cosine": 0.06144848234057426,
      "logmag_mse": 8.444940816345214,
      "logmag_r2": -61.44793464123857
    },
    "eval": {
      "n": 200000,
      "nll": -144.20111734375,
      "nll_per_dim": -1.126571229248047,
      "nll_baseline": -131.6650314501953,
      "nll_baseline_per_dim": -1.0286330582046508,
      "dnll": -12.536085893554713,
      "dnll_per_dim": -0.0979381710433962,
      "direction_cosine": 0.06089484272360802,
      "logmag_mse": 8.478618435821533,
      "logmag_r2": -66.95110473962805
    },
    "rollout": {
      "n_utterances": 16,
      "teacher_forced": {
        "n": 2422,
        "nll": -143.32205430305663,
        "nll_per_dim": -1.11970354924263,
        "nll_baseline": -129.82474189273194,
        "nll_baseline_per_dim": -1.0142557960369682,
        "dnll": -13.497312410324696,
        "dnll_per_dim": -0.10544775320566169,
        "direction_cosine": 0.05395398895863004,
        "logmag_mse": 8.549291019684057,
        "logmag_r2": -57.14647562396713
      },
      "rollout_context": {
        "n": 2422,
        "nll": 220454257625.61078,
        "nll_per_dim": 1722298887.7000842,
        "nll_baseline": -129.82474189273194,
        "nll_baseline_per_dim": -1.0142557960369682,
        "dnll": 220454257755.43552,
        "dnll_per_dim": 1722298888.71434,
        "direction_cosine": -0.006346086624233068,
        "logmag_mse": 9.777352330484941,
        "logmag_r2": -65.49891524835361
      },
      "gap_nll": 220454257768.93283,
      "gap_dnll": 220454257768.93283
    }
  }
]
phase1's tables.csv for phase3's zdyn:
[{"horizon_k":"1","slice":"all","train_n":"200000","train_nll":"-190.934343046875","train_nll_per_dim":"-1.491674555053711","train_nll_baseline":"-131.45609314941407","train_nll_baseline_per_dim":"-1.0270007277297974","train_dnll":"-59.47824989746093","train_dnll_per_dim":"-0.4646738273239135","eval_n":"200000","eval_nll":"-192.3472385546875","eval_nll_per_dim":"-1.502712801208496","eval_nll_baseline":"-131.2340589453125","eval_nll_baseline_per_dim":"-1.025266085510254","eval_dnll":"-61.11317960937498","eval_dnll_per_dim":"-0.47744671569824204","eval_direction_cos":"0.7084952416801452","eval_logmag_r2":"-0.09790971093190404","rollout_gap_nll":"68.6992258453369","rollout_gap_dnll":"68.6992258453369"},{"horizon_k":"2","slice":"all","train_n":"200000","train_nll":"-161.71146455078124","train_nll_per_dim":"-1.2633708168029785","train_nll_baseline":"-131.45609314941407","train_nll_baseline_per_dim":"-1.0270007277297974","train_dnll":"-30.25537140136717","train_dnll_per_dim":"-0.23637008907318102","eval_n":"200000","eval_nll":"-161.605119453125","eval_nll_per_dim":"-1.2625399957275392","eval_nll_baseline":"-131.2340589453125","eval_nll_baseline_per_dim":"-1.025266085510254","eval_dnll":"-30.3710605078125","eval_dnll_per_dim":"-0.23727391021728517","eval_direction_cos":"0.44011576730728147","eval_logmag_r2":"-5.733511868991406","rollout_gap_nll":"","rollout_gap_dnll":""},{"horizon_k":"4","slice":"all","train_n":"200000","train_nll":"-149.13276833984375","train_nll_per_dim":"-1.1650997526550293","train_nll_baseline":"-131.45609314941407","train_nll_baseline_per_dim":"-1.0270007277297974","train_dnll":"-17.676675190429677","train_dnll_per_dim":"-0.13809902492523185","eval_n":"200000","eval_nll":"-148.67317142578125","eval_nll_per_dim":"-1.161509151763916","eval_nll_baseline":"-131.2340589453125","eval_nll_baseline_per_dim":"-1.025266085510254","eval_dnll":"-17.439112480468737","eval_dnll_per_dim":"-0.136243066253662","eval_direction_cos":"0.1522323039674759","eval_logmag_r2":"-39.241701345516155","rollout_gap_nll":"","rollout_gap_dnll":""},{"horizon_k":"8","slice":"all","train_n":"200000","train_nll":"-144.47997451660157","train_nll_per_dim":"-1.1287498009109498","train_nll_baseline":"-131.91264184570312","train_nll_baseline_per_dim":"-1.0305675144195556","train_dnll":"-12.567332670898452","train_dnll_per_dim":"-0.09818228649139416","eval_n":"200000","eval_nll":"-144.20111734375","eval_nll_per_dim":"-1.126571229248047","eval_nll_baseline":"-131.6650314501953","eval_nll_baseline_per_dim":"-1.0286330582046508","eval_dnll":"-12.536085893554713","eval_dnll_per_dim":"-0.0979381710433962","eval_direction_cos":"0.06089484272360802","eval_logmag_r2":"-66.95110473962805","rollout_gap_nll":"220454257768.93283","rollout_gap_dnll":"220454257768.93283"}]

phase3 training log last line:
{"step": 39700, "kl_mode": "target", "beta": 0.0, "kl_target": 2.0, "kl_gamma": 0.5, "loss_total": 1.30083167552948, "loss_recon": 0.12094542384147644, "loss_kl": 0.0012460947036743164, "loss_dyn": 117.86402130126953, "kl_raw": 2.001953125, "z_dyn_var_mean": 0.009425630792975426, "z_dyn_l2_mean": 10.470450401306152, "z_rec_post_l2_mean": 4.785769462585449, "z_rec_prior_l2_mean": 6.068709373474121, "q_log_sigma_mean": 0.12139892578125, "p_log_sigma_mean": 0.304931640625, "z_dyn_delta_l2_mean": 1.01842200756073, "dyn_mse": 0.003748178482055664, "recon_post": 0.1209457740187645, "recon_prior": 0.1209447830915451, "recon_prior_over_post": 0.999991806847098, "prior_frac": 0.48968252539634705}
phase3 eval metrics:
{
  "step": 50000,
  "recon": 0.10968041643500329,
  "kl": 0.001731644868850708,
  "dyn": 117.83164840698242,
  "total": 1.2897285223007202,
  "n_batches": 50
}


phase4 metrics.json:
{
  "teacher_forced": [
    {
      "model": "memory",
      "n_batches": 200,
      "n_samples": 409600,
      "nll": 10.568189086914062,
      "nll_baseline": -131.7337088775635,
      "dnll": 142.30189796447755,
      "direction_cos": 0.4749230892956257
    },
    {
      "model": "param",
      "n_batches": 200,
      "n_samples": 409600,
      "nll": -176.0486181640625,
      "nll_baseline": -131.7337088775635,
      "dnll": -44.314909286499024,
      "direction_cos": 0.6926173835992813
    },
    {
      "model": "hybrid",
      "n_batches": 200,
      "n_samples": 409600,
      "nll": -177.1333562850952,
      "nll_baseline": -131.7337088775635,
      "dnll": -45.39964740753174,
      "direction_cos": 0.6972745847702027
    }
  ],
  "rollout": [
    {
      "model": "memory",
      "n_steps": 2723,
      "teacher_forced_nll": -154.12914050960646,
      "rollout_nll": -84.87703359683033,
      "teacher_forced_dnll": -23.253884637701656,
      "rollout_dnll": 45.99822227507447,
      "rollout_gap_nll": 69.25210691277613,
      "rollout_gap_dnll": 69.25210691277613
    },
    {
      "model": "param",
      "n_steps": 2723,
      "teacher_forced_nll": -176.7859268566724,
      "rollout_nll": NaN,
      "teacher_forced_dnll": -45.91067098476761,
      "rollout_dnll": NaN,
      "rollout_gap_nll": NaN,
      "rollout_gap_dnll": NaN
    },
    {
      "model": "hybrid",
      "n_steps": 2723,
      "teacher_forced_nll": -177.75988325650667,
      "rollout_nll": -108.73197446060321,
      "teacher_forced_dnll": -46.88462738460186,
      "rollout_dnll": 22.143281411301594,
      "rollout_gap_nll": 69.02790879590346,
      "rollout_gap_dnll": 69.02790879590346
    }
  ]
}