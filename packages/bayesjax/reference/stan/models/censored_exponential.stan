data {
  int<lower=1> N;
  int<lower=0> N_obs;
  int<lower=0> N_mis;
  array[N_obs] int<lower=1, upper=N> observed_idx;
  array[N_mis] int<lower=1, upper=N> missing_idx;
  vector<lower=0>[N_obs] observed_values;
  vector<lower=0>[N_mis] missing_lower;
  real<lower=0> prior_rate;
}
parameters {
  real<lower=0> rate;
  vector<lower=missing_lower>[N_mis] t_cens;
}
transformed parameters {
  vector[N] y_full;
  for (i in 1:N_obs) {
    y_full[observed_idx[i]] = observed_values[i];
  }
  for (i in 1:N_mis) {
    y_full[missing_idx[i]] = t_cens[i];
  }
}
model {
  rate ~ exponential(prior_rate);
  y_full ~ exponential(rate);
}
