########################################################################
# Samplers shared among all models
########################################################################

function update_basis_coefs_cov(y, X, sigma_inv, lambdas, total_preds, R)
    G = X' * sigma_inv * X + kron(Diagonal(lambdas), R) |> Hermitian;
    g = X' * sigma_inv * y;
    G_chol = cholesky(G);
    mean_est = G_chol.U \ (G_chol.L \ g);
    a = mean_est + (G_chol.U \ randn(total_preds));
    return a
end

function update_basis_coefs(y, X, tau, lambdas, total_preds, R)
    G = tau * X' * X + kron(Diagonal(lambdas), R) |> Hermitian;
    g = tau * X' * y;
    G_chol = cholesky(G);
    mean_est = G_chol.U \ (G_chol.L \ g);
    a = mean_est + (G_chol.U \ randn(total_preds));
    return a
end

function update_tau(errors, a_tau, b_tau, NT)
    this_shape = NT / 2 + a_tau; 
    this_rate = 1/2 * sum(errors.^2) + b_tau;
    return rand(Gamma(this_shape, 1.0 / this_rate)); 
end

function update_lambdas!(lambdas, A, R, a_lambda, b_lambda, M, P)
    for p in 1:P
        this_shape = a_lambda + M / 2.0;
        this_rate = b_lambda + 1/2 * A[:,p]' * R * A[:,p];
        lambdas[p] = rand(Gamma(this_shape, 1.0 / this_rate));
    end
end

# random effects smoothing parameter
function update_lambda_C(C, R, a_lambda, b_lambda, M, N)
    this_rate = b_lambda;
    this_shape = a_lambda + (M * N) / 2;
    for i in 1:N
        this_rate = this_rate + 1/2 * C[:,i]' * R * C[:,i];
    end
    lambda = rand(Gamma(this_shape, 1.0 / this_rate));
end



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# functions to update the factor model parameters
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
function update_Z(y, mgamma, lambdas_Z, cc_ind, tau, N, L)
    mgamma_tilde = kron(Diagonal(I, N), mgamma)[cc_ind, :];
    G = tau * mgamma_tilde' * mgamma_tilde + 
        kron(Diagonal(I, N), Diagonal(lambdas_Z));
    g = tau * mgamma_tilde' * y;
    G_chol = G |> Hermitian |> cholesky;
    mean_est = G_chol.U \ (G_chol.L \ g);
    z = mean_est + (G_chol.U \ randn(N * L));
    return z;
end


function update_C!(C, Z, lambdas_C, y, mtheta, R, mgamma, L, M, tau, cc_ind)
    l_index = sample(1:L, L, replace = false)
    ML = M * L;
    for l in l_index
        start = (l - 1) * M + 1; stop = l * M;
        samp_index = start:stop |> collect;
        error_index = setdiff(1:ML, samp_index);
        Z_tilde = kron(Z, mtheta)[cc_ind, :];
        ehat = y - Z_tilde[:,error_index] * vec(C)[error_index];
        Z_tilde = Z_tilde[:, samp_index];
        G = tau * Z_tilde' * Z_tilde + lambdas_C[l] * R;
        g = tau * Z_tilde' * ehat;
        F = mtheta' * mgamma[:, 1:end .!= l];
        G_chol = G |> Hermitian |> cholesky;
        c_0 = G_chol.U \ ((G_chol.L \ g) + randn(M));
        F_tilde = G_chol.U \ (G_chol.L \ F);
        c_l = c_0 - F_tilde * inv(F' * F_tilde) * F' * c_0
        norm_mgamma_l = norm(mtheta * c_l);
        C[:, l] = c_l / norm_mgamma_l;
        mgamma[:, l] = mtheta * C[:, l];
        Z[:, l] = norm_mgamma_l * Z[:,l];
    end
end


function update_lambdas_Z(lambdas_C, C, a_prior, b_prior, M, L)
    for l in 1:L
        shape = M/2 + a_prior;
        rate = 1/2 * C[:,l]' * C[:,l] + b_prior;
        scale = 1.0 / rate;
        lambdas_Z[l] = rand(Gamma(shape, scale));
    end
end

#**********************************************************************#
# rearrange cluster indicies to 1:K
#**********************************************************************#
function rearrange_clust_info!(d_ind, this_K, temp_lambdas, lambdas_B, 
                               clust_names)

    if maximum(clust_names) > length(lambdas_B)
        println("this will break 3")
    end

    for k in 1:this_K
        d_ind[d_ind .== clust_names[k]] .= k;
        temp_lambdas[k] = lambdas_B[clust_names[k]];
    end
end

#**********************************************************************#
# some other helpers
#**********************************************************************#
# create auxillary lambdas
function create_aux_lambda(temp_lambdas, lambdas_B, is_alone, d_ind, 
                           j, a_lambda, b_lambda);
    if d_ind[j] == 0
        new_lambda = rand(Gamma(a_lambda, 1/b_lambda));
        lambdas_B = [temp_lambdas; new_lambda];
    elseif is_alone
        lambdas_B = [temp_lambdas; lambdas_B[d_ind[j]]];
    else
        new_lambda = rand(Gamma(a_lambda, 1/b_lambda));
        lambdas_B = [temp_lambdas; new_lambda];
    end
   return lambdas_B;
end

# get the posterior probabilities of a cluster indicator
function get_clust_post(prior_probs, log_lik)
    post_probs_log = log.(prior_probs) + log_lik;
    post_probs_log = post_probs_log .- maximum(post_probs_log);
    post_probs = Weights(exp.(post_probs_log) |> normalize_probs);
    return post_probs
end

#**********************************************************************#
# update the eta
#**********************************************************************#
function update_eta(P, K, a_prior, b_prior, eta)

    # auxillary variable
    x = rand(Beta(eta + 1, P));
    
    a_star = (a_prior + K - 1);
    b_star = P * (b_prior - log(x));
    pi_x = a_star / (a_star + b_star);

    mix_ind = rand(Bernoulli(pi_x));

    if mix_ind == 1
        this_shape = a_prior + K;
    else
        this_shape = a_prior + K - 1;
    end

    this_scale = 1.0 / (b_prior - log(x));
    eta = rand(Gamma(this_shape, this_scale));

    return eta
end
