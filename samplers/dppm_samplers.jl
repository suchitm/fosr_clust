#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# update the cluster indicator variables 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
function update_d_ind_dppm!(d_ind, P_c, X, ehat, lambdas_B, tau, eta, eta_0, 
                            a_lambda, b_lambda, mtheta, R, NT, M, cc_ind)

    iter_seq = sample(1:P_c, P_c, replace = false);

    for j in iter_seq

        prior_probs, lambdas_B, this_K = get_prior_and_lambdas_dppm!(
            j, d_ind, eta_0, eta, P_c, a_lambda, b_lambda, lambdas_B);
        
        # get the log likelihood under each cluster
        log_lik = Array{Float64, 1}(undef, this_K + 2);
        get_log_lik_dppm!(log_lik, d_ind, j, X, ehat, tau, lambdas_B, 
                          mtheta, R, NT, M, this_K, cc_ind)
        
        # normalize the posterior probs and sample new cluster membership
        post_probs = get_clust_post(prior_probs, log_lik)
        d_ind[j] = sample(0:(this_K+1), post_probs)

        # rearranging again; need to account for if all is null
        is_null = (d_ind .== 0) * 1;
        temp_d_ind = d_ind[(is_null .== 0)];
        
        if temp_d_ind == Int64[]
            lambdas_B = []
        else
            clust_names = temp_d_ind |> countmap |> sort |> keys |> collect;
            this_K = length(clust_names);
            temp_lambdas = Array{Float64}(undef, this_K);
            rearrange_clust_info!(d_ind, this_K, temp_lambdas, lambdas_B, 
                                  clust_names)
            lambdas_B = temp_lambdas
        end
    end

    return lambdas_B

end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# get log likelihood via integration 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
function get_log_lik_dppm!(log_lik, d_ind, j, X, ehat, tau, lambdas_B, 
                           mtheta, R, NT, M, this_K, cc_ind)

    for k in 0:(this_K + 1)
        
        # set cluster indicator and find out how many are on-zero
        d_ind[j] = k;
        temp_d_ind = d_ind[d_ind .!= 0];

        # if everyone zero then log-lik is a lot simpler
        if temp_d_ind == Int64[]
            log_lik[k+1] = 
                NT / 2 * log(tau) -
                1 / 2 * (tau * ehat' * ehat)
            continue
        end
    
        # the case where ther are non-null predictors
        # getting the vecc-ed predictor matrix
        n_clust = temp_d_ind |> countmap |> length;
        D = make_D(temp_d_ind, n_clust);
        X_sub = X[:, d_ind .!= 0];
        X_tilde = kron(X_sub * D, mtheta)[cc_ind, :];
 
        # which lambdas correspond to the clusters?
        if k == 0
            temp_k = 1;
            temp_lambdas = lambdas_B[[1:n_clust; temp_k] |> unique |> sort];
        else 
            temp_lambdas = lambdas_B[[1:n_clust; k] |> unique |> sort];
        end

        # precision matrix and cholesky
        G = X_tilde' * X_tilde + 1 / tau * kron(Diagonal(temp_lambdas), R);
        g = X_tilde' * ehat;
        G_chol = G |> Hermitian |> cholesky;
        gt_Ginv_g = sum((G_chol.L \ g).^2);
        
        # log likelihood
        log_lik[k+1] = 
            -NT / 2 * log(2 * pi) + 
            (NT - M * n_clust) / 2 * log(tau) + 
            M / 2 * sum(log.(temp_lambdas)) + 
            n_clust / 2 * logdet(R) - 
            1/2 * logdet(G_chol) - 
            tau/2 * (ehat' * ehat - gt_Ginv_g);

    end
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# functions to get the prior probabilities of 
# cluster membership
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# combines the null and the clust prior
function get_clust_prior_dppm(d_ind, is_null, eta, eta_0, P_c, j)
    null_prior = get_null_clust_prior(is_null, j, eta_0, P_c);
    clust_prior = get_act_clust_prior(d_ind, is_null, j, eta);
    # the second entry in the null-prior corresponds to the probability
    # of a variable being null
    prior_probs = [null_prior[2]; clust_prior * null_prior[1]];
    return prior_probs
end

# get the prior probs for the null cluster
function get_null_clust_prior(is_null, j, eta_0, P_c)
    null_dict = is_null[j .!= 1:end] |> countmap |> sort;
    null_keys = null_dict |> keys |> collect;
    # all 0 or all 1 need to adjust
    if null_keys != [0; 1]
        if null_keys[1] == 0
            null_dict[:1] = 0;
        else
            null_dict[:0] = 0;
        end
        null_dict = null_dict |> sort;
    end
    null_clust_values = null_dict |> values |> collect;
    null_prior = (null_clust_values .+ eta_0 / 2) / (P_c - 1 + eta_0);
    return null_prior
end

# getting prior for the variables active for DP process prior
function get_act_clust_prior(d_ind, is_null, j, eta)
    temp_P = sum(is_null[j .!= 1:end] .== 0) + 1;
    clust_table_counts = d_ind[(j .!= 1:end) .& (d_ind .!= 0)] |> 
        countmap |> sort |> values |> collect;
    clust_prior = clust_table_counts / (temp_P - 1 + eta);
    clust_prior = [clust_prior; eta / (temp_P - 1 + eta)];
    return clust_prior
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# function to take into account that everything can be null
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
function get_prior_and_lambdas_dppm!(j, d_ind, eta_0, eta, P_c, a_lambda, 
                                     b_lambda, lambdas_B)

    # determining the null clusters and the clusters which are non-zero
    is_null = (d_ind .== 0) * 1;
    temp_d_ind = d_ind[(is_null .== 0) .& (1:end .!= j)];

    # it is possible for all predictors to be set to zero so we have
    # to deal with that in the prior probability calculation
    if temp_d_ind == Int64[]
        # the prior if everything is null is the 2-component mixture 
        # model prior; the proposal is that the element will not be null
        this_K = 0;
        prior_probs = get_null_clust_prior(is_null, j, eta_0, P_c);
        prior_probs = [prior_probs[2], prior_probs[1]];

        #creating an auxillary lambda
        temp_lambdas = Array{Float64}(undef, this_K);
        is_alone = true;
        lambdas_B = create_aux_lambda(temp_lambdas, lambdas_B, is_alone, 
                                      d_ind, j, a_lambda, b_lambda);

        return prior_probs, lambdas_B, this_K;
    end

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # the case where ther are non-null predictors
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # setup for prior probs
    clust_dict = temp_d_ind |> countmap |> sort;
    clust_names = clust_dict |> keys |> collect;
    is_alone = ((d_ind[j] .== temp_d_ind) |> sum) == 0;

    # rearrange the cluster indicators and lambdas
    this_K = length(clust_names);
    temp_lambdas = Array{Float64}(undef, this_K);
    rearrange_clust_info!(d_ind, this_K, temp_lambdas, lambdas_B, clust_names)

    #creating an auxillary lambda
    lambdas_B = create_aux_lambda(temp_lambdas, lambdas_B, is_alone, d_ind, j, 
                                  a_lambda, b_lambda)

    # get the prior probabilities
    prior_probs = get_clust_prior_dppm(d_ind, is_null, eta, eta_0, P_c, j)

    return prior_probs, lambdas_B, this_K;
end
