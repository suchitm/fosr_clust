function fosr_dp(Y, W, X, mtheta, R, n_iter, verbose = true)
    
    N, P_f = size(W);
    P_c = size(X, 2);
    P = P_f + P_c;
    T, M = size(mtheta);

    y = vec(Y');
    cc_ind = .!(ismissing.(y));
    y = y[cc_ind];

    W_tilde = kron(W, mtheta)[cc_ind, :];
    NT = length(y);
    free_coefs = M * P_f;
    loading_coefs = M * N;

    # prior for tau
    a_tau = b_tau = a_eta = b_eta = 0.01;
    # prior for the lambdas 
    a_lambda = 0.1;
    b_lambda = 0.1;

    # matricies to store results
    funcs_array = Array{Float64, 3}(undef, T, P, n_iter);
    tau_vec = Float64[];
    d_mat = Array{Int16, 2}(undef, P_c, n_iter);
    eta_vec = Float64[];

    # starting values for the parameters
    # fosr_dp starters
    tau = 1; eta = 1.0;
    d_ind = sample(1:1, P_c, replace = true);
    a = randn(free_coefs);
    lambdas_A = repeat([1.0], inner = P_f);
    lambdas_B = repeat([1.0], inner = d_ind |> unique |> length);

    for i in 1:n_iter

        if verbose
            if (i % 500) == 0
                print(i, '\r')
            end
        end

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # updating the parameters for the clusters 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # update cluster indicators
        ehat = y - W_tilde * a;
        lambdas_B = update_d_ind_dp!(d_ind, P_c, X, ehat, lambdas_B, tau, eta, 
                                      a_lambda, b_lambda, mtheta, R, NT, M, 
                                      cc_ind);

        # information about how many coefs
        K = d_ind |> unique |> length;
        clust_coefs = M * K;
        D = make_D(d_ind, K);

        # update the mean parameters for the clusters
        X_tilde = kron(X * D, mtheta)[cc_ind,:];
        b = update_basis_coefs(ehat, X_tilde, tau, lambdas_B, clust_coefs, R);
        B = reshape(b, M, K);
        # update cluster smoothing
        update_lambdas!(lambdas_B, B, R, a_lambda, b_lambda, M, K);

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # update coefficients for control variables
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # mean
        ehat = y - X_tilde * b;
        a = update_basis_coefs(ehat, W_tilde, tau, lambdas_A, free_coefs, R);
        A = reshape(a, M, P_f);
        # smoothing parameters
        update_lambdas!(lambdas_A, A, R, a_lambda, b_lambda, M, P_f);
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # updating tau
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        errors = y - W_tilde * a - X_tilde * b;
        tau = update_tau(errors, a_tau, b_tau, NT);
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # update eta
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        eta = update_eta(eta, P_c, K, a_eta, b_eta);
        
        # storing the restuls
        push!(tau_vec, tau);
        push!(eta_vec, eta);
        funcs_array[:,:,i] = hcat(mtheta * A, mtheta * B * D');
        d_mat[:, i] = d_ind;

    end

    return funcs_array, d_mat, tau_vec, eta_vec; 
end
