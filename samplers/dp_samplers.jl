#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# update the cluster indicator variable by 
# integrating the basis function coefficients
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
function update_d_ind_dp!(d_ind, P_c, X, ehat, lambdas_B, tau, eta, 
                          a_lambda, b_lambda, mtheta, R, NT, M, cc_ind)
    
    iter_seq = sample(1:P_c, P_c, replace = false);

    for j in iter_seq

        # rearranging the cluster index, K, and is alone
        clust_dict = d_ind[1:end .!= j] |> countmap |> sort;
        clust_names = clust_dict |> keys |> collect;
        is_alone = (d_ind[j] .== d_ind[1:end .!= j]) |> sum == 0;
        
        # rearrange the cluster indicators and lambdas
        this_K = length(clust_names);
        temp_lambdas = Array{Float64}(undef, this_K);
        rearrange_clust_info!(d_ind, this_K, temp_lambdas, lambdas_B, 
                              clust_names)

        #creating an auxillary lambda
        lambdas_B = create_aux_lambda(
            temp_lambdas, lambdas_B, is_alone, d_ind, j, a_lambda, 
            b_lambda);
        
        # get the prior probabilities
        prior_probs = get_clust_prior_dp(d_ind, j, eta);
        
        # get the log likelihood under each cluster
        log_lik = Array{Float64, 1}(undef, this_K + 1);
        get_log_lik_dp!(log_lik, d_ind, j, X, ehat, tau, lambdas_B, mtheta, 
                         R, NT, M, this_K, cc_ind)
        
        # normalize the posterior probs and get new index
        post_probs = get_clust_post(prior_probs, log_lik);

        # sampling the cluster indicator
        d_ind[j] = sample(1:(this_K+1), post_probs);

        # rearranging again
        clust_names = d_ind |> countmap |> sort |> keys |> collect;
        this_K = length(clust_names);
        temp_lambdas = Array{Float64}(undef, this_K);
        rearrange_clust_info!(d_ind, this_K, temp_lambdas, lambdas_B, 
                              clust_names)
        lambdas_B = temp_lambdas;

    end
    return lambdas_B
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# get log likelihood via integration 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
function get_log_lik_dp!(log_lik, d_ind, j, X, ehat, tau, lambdas_B, mtheta, 
                         R, NT, M, this_K, cc_ind)
    for k in 1:(this_K + 1)
        # make the cluster indicator matrix and vec covariate matrix
        d_ind[j] = k;
        n_clust = d_ind |> countmap |> length;
        D = make_D(d_ind, n_clust);
        X_tilde = kron(X * D, mtheta)[cc_ind,:];
        temp_lambdas = lambdas_B[[1:n_clust; k] |> unique |> sort];
        # cholesky for the precision matrix
        G = X_tilde' * X_tilde + 1 / tau * kron(Diagonal(temp_lambdas), R);
        g = X_tilde' * ehat; 
        G_chol = G |> Hermitian |> cholesky;
        gt_Ginv_g = sum((G_chol.L \ g).^2);

        # log likelihood
        log_lik[k] = 
            -NT / 2 * log(2 * pi) + 
            (NT - M * n_clust) / 2 * log(tau) + 
            M / 2 * sum(log.(temp_lambdas)) + 
            n_clust / 2 * logdet(R) - 
            1/2 * logdet(G_chol) - 
            tau/2 * (ehat' * ehat - gt_Ginv_g);

    end
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# prior probs for the cluster indicators
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
function get_clust_prior_dp(d_ind, j, eta)
    P = length(d_ind);
    clust_dict = d_ind[1:end .!= j] |> countmap |> sort;
    clust_counts = clust_dict |> values |> collect;
    prior_probs = [
        clust_counts / (P - 1.0 + eta); 
        eta / (P - 1.0 + eta)
       ];
    return prior_probs
end