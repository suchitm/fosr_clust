function run_one_simulation(Y, W, X, mtheta, R, n_iter, burn_in, model_name, 
                            true_funcs, true_d_ind, true_cov_mat)

    T, M = size(mtheta); 
    seq_to_keep = Array{Int64}(burn_in:n_iter);

    if model_name == "fosr"

        W_new = hcat(W, X);
        funcs_array, tau_vec = fosr(Y, W_new, mtheta, R, n_iter);
        adj_rand = 0.0;
        rand_index = 0.0;
        estim_cov_mat = 1 / mean(tau_vec[seq_to_keep]) * Diagonal{Float64}(I, T)

    elseif model_name == "fosr_pm"
        
        funcs_array, d_mat, tau_vec = fosr_pm(Y, W, X, mtheta, R, n_iter);
        adj_rand = 0.0;
        rand_index = 0.0;
        estim_cov_mat = 1 / mean(tau_vec[seq_to_keep]) * Diagonal{Float64}(I, T)

    elseif model_name == "fosr_dp"

        funcs_array, d_mat, tau_vec, eta_vec = fosr_dp(Y, W, X, mtheta, R, n_iter);
        estim_d_ind = best_d_ind(d_mat[:, seq_to_keep])
        rand_index, adj_rand = get_rand_ind(true_d_ind, estim_d_ind)
        estim_cov_mat = 1 / mean(tau_vec[seq_to_keep]) * Diagonal{Float64}(I, T)

    elseif model_name == "fosr_dppm"

        funcs_array, d_mat, tau_vec, eta_vec = fosr_dppm(Y, W, X, mtheta, R, n_iter);
        estim_d_ind = best_d_ind(d_mat[:, seq_to_keep])
        estim_d_ind = estim_d_ind .+ 1; # adjusts if everything is zero
        rand_index, adj_rand = get_rand_ind(true_d_ind, estim_d_ind)
        estim_cov_mat = 1 / mean(tau_vec[seq_to_keep]) * Diagonal{Float64}(I, T)

    end
    
    coverage, mse_funcs = calculate_mse(true_funcs, funcs_array, seq_to_keep) 
    mse_cov = mean((estim_cov_mat - true_cov_mat).^2)
    
    return (coverage = coverage, mse_funcs = mse_funcs, 
            arand = adj_rand, rand = rand_index, mse_cov = mse_cov)

end

########################################################################
# Functions to generate data for the simulation by design
########################################################################

function gen_sim_data(design, N, rsnr)
    if design == 1
        X, W, Y, true_d_ind, true_funcs, true_cov_mat, mtheta, R =
            gen_design_1(N, rsnr)
    elseif design == 2
        X, W, Y, true_d_ind, true_funcs, true_cov_mat, mtheta, R =
            gen_design_2(N, rsnr)
    elseif design == 3
        X, W, Y, true_d_ind, true_funcs, true_cov_mat, mtheta, R =
            gen_design_3(N, rsnr)
    elseif design == 4
        X, W, Y, true_d_ind, true_funcs, true_cov_mat, mtheta, R =
            gen_design_4(N, rsnr)
    end
    return X, W, Y, true_d_ind, true_funcs, true_cov_mat, mtheta, R
end


#~~~~~~~~~~~~~~~~~~~~~~~~~#
# Design 1
#~~~~~~~~~~~~~~~~~~~~~~~~~#
function gen_design_1(N, rsnr)
    # setup
    P_f = 5;
    P_c = 15; 
    T = 15;
    # cluster inds
    true_d_ind = [repeat([1], inner = 7); repeat([2; 3], inner = 4)];;
    # generating and returning data
    X, W, Y, true_d_ind, true_funcs, true_cov_mat, mtheta, R = 
        get_data(N, rsnr, P_f, P_c, T, true_d_ind, true)
    return X, W, Y, true_d_ind, true_funcs, true_cov_mat, mtheta, R
end

#~~~~~~~~~~~~~~~~~~~~~~~~~#
# Design 2
#~~~~~~~~~~~~~~~~~~~~~~~~~#
function gen_design_2(N, rsnr)
    # setup
    P_f = 5;
    P_c = 15; 
    T = 15;
    # cluster inds
    true_d_ind = [repeat([1], inner = 7); repeat([2; 3], inner = 4)];;
    # generating and returning data
    X, W, Y, true_d_ind, true_funcs, true_cov_mat, mtheta, R = 
        get_data(N, rsnr, P_f, P_c, T, true_d_ind, false)
    return X, W, Y, true_d_ind, true_funcs, true_cov_mat, mtheta, R
end

#~~~~~~~~~~~~~~~~~~~~~~~~~#
# Design 3
#~~~~~~~~~~~~~~~~~~~~~~~~~#
function gen_design_3(N, rsnr)
    # setup
    P_f = 5;
    P_c = 15; 
    T = 15;
    # cluster inds
    true_d_ind = 1:P_c |> collect;
    # generating and returning data
    X, W, Y, true_d_ind, true_funcs, true_cov_mat, mtheta, R = 
        get_data(N, rsnr, P_f, P_c, T, true_d_ind, false)
    return X, W, Y, true_d_ind, true_funcs, true_cov_mat, mtheta, R
end

#~~~~~~~~~~~~~~~~~~~~~~~~~#
# Design 4
#~~~~~~~~~~~~~~~~~~~~~~~~~#
function gen_design_4(N, rsnr)
    # setup
    P_f = 5;
    P_c = 15; 
    T = 15;
    # cluster indicators
    true_d_ind = sample(1:1, P_c);
    true_d_ind[8:end] = 2:9 |> collect;
    # generating and returning data
    X, W, Y, true_d_ind, true_funcs, true_cov_mat, mtheta, R = 
        get_data(N, rsnr, P_f, P_c, T, true_d_ind, true)
    return X, W, Y, true_d_ind, true_funcs, true_cov_mat, mtheta, R
end

#~~~~~~~~~~~~~~~~~~~~~~~~~#
# masterr data gen func
#~~~~~~~~~~~~~~~~~~~~~~~~~#
function get_data(N, rsnr, P_f, P_c, T, true_d_ind, is_zero = true)
    
    fourier_bases = 3;
    bspline_bases = 8; M = bspline_bases; 
    time_eval = range(0, stop = 1, length = T) |> collect;
    fourier_mat, mtheta = get_basis_mats(N, T, fourier_bases, bspline_bases, 
                                            time_eval);
    diff_mat = get_diff_mat(2, M);
    R = 0.001 * Diagonal{Float64}(I, M) + diff_mat' * diff_mat;

    # generating W and the corresponding coefficient function
    W = randn(N, P_f)
    W[:,1] .= 1;
    A = randn(fourier_bases, P_f);
    alpha_mat = fourier_mat * A;

    # generating X with column values correlated
    X_cov_mat = Array{Float64, 2}(undef, P_c, P_c)
    for i in 1:P_c
        for j in 1:P_c
            X_cov_mat[i, j] = 0.75^(abs(i - j))
        end
    end
    X_cov_mat = Hermitian(X_cov_mat);
    X = cholesky(X_cov_mat).L * randn(P_c, N);
    X = X';

    # d_ind, and the corresponding clustered functions
    K = true_d_ind |> unique |> length;
    D = make_D(true_d_ind, K);
    B = randn(fourier_bases, K);
    if is_zero 
        B[:,1] .= 0;
    end
    beta_mat = fourier_mat * B * D';
    
    true_funcs = hcat(alpha_mat, beta_mat);

    # GP covariance function
    cov_mat = Array{Float64, 2}(undef, T, T);
    for i in 1:T
        for j in 1:T
            cov_mat[i, j] = exp(- 10 * (time_eval[i] - time_eval[j])^2)
        end
    end
    # determining the nugget for the data
    Y_star = W * alpha_mat' + X * beta_mat';
    sigma2 = max(0, var(Y_star) / rsnr^2 - 1);
    # final covariance
    true_cov_mat = cov_mat + sigma2 * Diagonal{Float64}(I, T)
    
    #Y_star = W * alpha_mat' + X * beta_mat';
    #sigma2 = var(Y_star) / rsnr^2;
    #true_cov_mat = sigma2 * Diagonal{Float64}(I, T);

    cov_chol = cholesky(true_cov_mat)
    # generating the data
    Y = Y_star;
    for n in 1:N
        Y[n, :] = Y_star[n, :] + cov_chol.L * randn(T);
    end

    return X, W, Y, true_d_ind, true_funcs, true_cov_mat, mtheta, R

end

#**********************************************************************#
# other random helpers
#**********************************************************************#
function get_time_index(N, T, prop_start, prop_end)
    time_index = Array{ Array{Int64, 1}, 1}(undef, N);
    miss_index = Array{ Array{Int64, 1}, 1}(undef, N);
    for i in 1:N
        prop = rand(Uniform(prop_start, prop_end));
        n_samps = Int64(round(T * prop, digits = 0));
        time_index[i] = [1; sample(2:T, n_samps - 1, replace = false)] |> sort;
        miss_index[i] = setdiff(1:T, time_index[i]);
    end
    return time_index, miss_index
end

function get_basis_mats(N, T, M, L, time_eval)
    start = minimum(time_eval);
    stop = maximum(time_eval);
    @rput T; @rput N; @rput M; @rput L; @rput start; @rput stop;
    @rput time_eval;
    R"""
    library(fda)
    library(dplyr) 
    fourier_basis = create.fourier.basis(
        rangeval = c(start, stop), 
        nbasis = M
    )
    fourier_mat = eval.basis(
        evalarg = seq(start, stop, length = T), 
        basisobj = fourier_basis
    )
    bspline_basis = create.bspline.basis(
        rangeval = c(start, stop),
        nbasis = L
    )
    theta_mat = eval.basis(
        evalarg = seq(start, stop, length = T),
        basisobj = bspline_basis
    )
    """;
    @rget fourier_mat; @rget theta_mat;
    return fourier_mat, theta_mat;
end

function get_diff_mat(k, n)
    D = diff(Diagonal{Float64}(I, n), dims = 1);
    if k == 1
        return D
    end
    for i in 1:(k-1)
        D = diff(Diagonal{Float64}(I, n - i), dims = 1) * D;
    end
    return D
end

function make_D(d_ind, K, P_free = 0)
    K_temp = K + P_free;
    full_d_ind = [1:P_free; d_ind .+ P_free];
    N = length(full_d_ind)
    D = zeros(Int8, N, K_temp);
    for i = 1:N
        D[i, full_d_ind[i]] = 1;
    end
    return D
end

function normalize_probs(x)
    x / sum(x)
end

#***************************************************************************#
# function to calculate coverage and mse for the betas
#***************************************************************************#
function calculate_mse(true_funcs, funcs_array, seq_to_keep)

    T, P = size(true_funcs);
    func_quants = Array{Float64, 3}(undef, T, P, 2);

    for p in 1:P
        for t in 1:T
            func_quants[t, p, :] = 
                quantile(funcs_array[t, p, seq_to_keep], [0.025, 0.975]);
        end
    end

    coverage = ((true_funcs .<= func_quants[:,:,2]) .& 
        (true_funcs .>= func_quants[:,:,1])) |> mean;

    estim = median(funcs_array[:,:,seq_to_keep], dims = 3)[:,:,1];
    mse = mean((estim - true_funcs).^2);

    return coverage, mse; 
end

#**********************************************************************#
# Functions to estimate the cluster membership using Dahl (2006)
# Kim, Tadesse et. al. (2006)
#**********************************************************************#

function best_d_ind(d_ind)
    P_c, N = size(d_ind)
    freq_mat = make_freq_mat(d_ind)
    loss = Array{Float64, 1}(undef, length(seq_to_keep))
    for i in 1:N
        adj_mat = make_adj_mat(d_ind[:, i], P_c)
        loss[i] = abs.(adj_mat - freq_mat) |> sum
    end
    return d_ind[:, argmin(loss)]
end

function make_freq_mat(d_ind)
    P_c, N = size(d_ind);
    freq_matrix = Array{Float64}(undef, P_c, P_c);
    freq_matrix .= 0.0
    for i in 1:N
        for j in 1:P_c
            for l in 1:P_c
                if d_ind[j, i] == d_ind[l, i];
                    freq_matrix[j, l] += 1;
                end
            end
        end
    end
    freq_matrix = freq_matrix ./ length(seq_to_keep)
    return freq_matrix
end

function make_adj_mat(this_d_ind, P_c)
    adj_mat = Array{Int64}(undef, P_c, P_c);
    adj_mat .= 0;
    for i in 1:P_c
        for j in 1:P_c
            if this_d_ind[i] == this_d_ind[j]
                adj_mat[i, j] = 1;
            end
        end
    end
    return adj_mat
end

# get the clusterings based on the variable names
function get_clust_dict(estim_d_ind, var_names, zero)
    clust_info = Dict()
    n_clust = estim_d_ind |> unique |> length;
    if zero 
        seq_iter = 0:(n_clust - 1);
    else
        seq_iter = 1:n_clust
    end
    for i in seq_iter
        this_clust = var_names[estim_d_ind .== i];
        clust_info["clust_$(i)"] = this_clust
    end
    return clust_info
end
 
function get_estim_funcs(funcs_array, seq_to_keep)
    T, P = size(funcs_array[:,:,1]);
    func_quants = Array{Float64, 3}(undef, T, P, 2);
    for p in 1:P
        for t in 1:T
            func_quants[t, p, :] = 
                quantile(funcs_array[t, p, seq_to_keep], [0.025, 0.975]);
        end
    end
    estim = median(funcs_array[:,:,seq_to_keep], dims = 3)[:,:,1];
    return estim, func_quants[:,:,1], func_quants[:,:,2];
end

#**********************************************************************#
# Generate wishart from a cholesky factor
#**********************************************************************#
function gen_wish(nu, L, T)
    A = zeros(size(L))
    for i in 1:T
        A[i, i] = rand(Chi(nu - i + 1.0));
    end
    for j in 1:T-1, i in j+1:T
        A[i, j] = randn()
    end
    return L * A * A' * L'
end

function gp_cov(T)
    time_eval = range(0, stop = 1, length = T) |> collect;
    # GP covariance function
    cov_mat = Array{Float64, 2}(undef, T, T);
    for i in 1:T
        for j in 1:T
            cov_mat[i, j] = exp(- 10 * (time_eval[i] - time_eval[j])^2)
        end
    end
    return cov_mat
end

function get_rand_ind(c1, c2)
    @rput c1; @rput c2;
    R"""
    library(fossil)
    rand_ind = rand.index(c1, c2)
    adj_rand = adj.rand.index(c1, c2)
    """
    @rget rand_ind; @rget adj_rand;
    return rand_ind, adj_rand;
end

#**********************************************************************#
# create "sparseness" in the design matrix
#**********************************************************************#

function sparsify_Y(Y, prop_miss = 0.5)
    n, t = size(Y)
    Ym = Array{Union{Missing, Float64}, 2}(missing, n, t)
    for i = 1:n
        sparse_index = sample(1:t, Int64(floor(t * prop_miss)), replace = false)
        Ym[i, sparse_index] = Y[i, sparse_index]
    end
    Y = Ym
    return Y
end
