function fosr_dppm(Y, W, X, mtheta, R, n_iter, verbose = true)
    
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

    # prior for precison params
    a_tau = b_tau = 0.1; 
    a_lambda = b_lambda = 0.1;
    # eta is the dirichlet process concentration parameter
    a_eta = b_eta = 0.1;

    # matricies to store results
    funcs_array = Array{Float64, 3}(undef, T, P, n_iter);
    tau_vec = Float64[];
    eta_vec = Float64[];
    d_mat = Array{Int16, 2}(undef, P_c, n_iter);
    
    # starting values for the parameters
    # fosr dppm starters
    tau = 1; eta = 1.0; eta_0 = 2.0;
    d_ind = sample(0:1, P_c, replace = true);
    K = d_ind[d_ind .!= 0] |> unique |> length;
    a = randn(free_coefs);
    lambdas_A = repeat([1.0], inner = P_f);
    lambdas_B = repeat([1.0], inner = K);
    
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
        ehat = ehat[:,1];
        lambdas_B = update_d_ind_dppm!(d_ind, P_c, X, ehat, lambdas_B, tau, 
                                       eta, eta_0, a_lambda, b_lambda, 
                                       mtheta, R, NT, M, cc_ind)
        
        # info from the cluster update
        temp_d_ind = d_ind[d_ind .!= 0];
        K_nz = temp_d_ind |> unique |> length;
        K = K_nz + 1;

        if temp_d_ind == Int64[]
            # if everything is zero then B has only one column
            B = Array{Float64}(undef, M, K);
            B .= 0.0;
            # this makes sure that X_tilde * b = 0;
            X_tilde = zeros(NT, 1); b = 0.0;
        else
            # subset to non-zeros (nz) and make new predictor matrix
            X_sub = X[:, d_ind .!= 0];
            clust_coefs = M * K_nz;
            D = make_D(temp_d_ind, K_nz);
            # update the mean parameters for the clusters
            X_tilde = kron(X_sub * D, mtheta)[cc_ind,:];
            b = update_basis_coefs(ehat, X_tilde, tau, lambdas_B, clust_coefs, 
                                   R);
            B = reshape(b, M, K_nz);
            # update cluster smoothing
            update_lambdas!(lambdas_B, B, R, a_lambda, b_lambda, M, K_nz);
            # adding a zero column to B
            B = hcat(zeros(M), B)
        end

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # update coefficients for control variables
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # mean
        ehat = y - X_tilde * b;
        ehat = ehat[:,1];
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
        temp_P_c = length(temp_d_ind);
        if temp_P_c > 0
            eta = update_eta(eta, temp_P_c, K_nz, a_eta, b_eta);
        else 
            eta = rand(Gamma(a_eta, 1 / b_eta));
        end
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # storing the restuls
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        D = make_D(d_ind .+ 1, K);
        push!(tau_vec, tau);
        push!(eta_vec, eta);
        funcs_array[:,:,i] = hcat(mtheta * A, mtheta * B * D');
        d_mat[:, i] = d_ind;

    end

    return funcs_array, d_mat, tau_vec, eta_vec;
end
