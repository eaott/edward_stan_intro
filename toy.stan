functions {
  matrix network(matrix x, matrix w1, row_vector b1, matrix w2, row_vector b2) {
    {
      matrix[rows(x), 100] z;
      z = x * w1 + rep_matrix(b1, rows(x));
      for (j in 1:100) {
        for (i in 1:rows(x)) {
          z[i, j] = fmax(z[i, j], 0.0);
        }
      }
      return z * w2 + rep_matrix(b2, rows(x));
    }
  }
}

data {
    int<lower=0> N; // number of data points
    real x[N]; // inputs -- consider using vectors instead
    real y[N]; // outputs

    int<lower=0> N_test;
    real x_test[N_test];
}

parameters {
    real<lower=0> lambda;
    real<lower=0> gamma;
    row_vector[100] b1;
    matrix[1, 100] w1;
    row_vector[1] b2;
    matrix[100, 1] w2;
}


model {
    lambda ~ gamma(6.0, 6.0);
    gamma ~ gamma(6.0, 6.0);
    b1 ~ normal(0.0, 1 / sqrt(lambda));
    to_vector(w1) ~ normal(0.0, 1 / sqrt(lambda));
    b2 ~ normal(0.0, 1 / sqrt(lambda));
    to_vector(w2) ~ normal(0.0, 1 / sqrt(lambda));

    y ~ normal(to_vector(network(to_matrix(x, N, 1), w1, b1, w2, b2)), 1 / sqrt(gamma));
}

generated quantities {
  matrix[N_test, 1] y_test;
  {
    matrix[N_test, 1] out;
    out = network(to_matrix(x_test, N_test, 1), w1, b1, w2, b2);
    for (i in 1:N_test) {
      y_test[i, 1] = normal_rng(out[i, 1], 1 / sqrt(gamma));
    }
  }
}
