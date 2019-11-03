using StatsBase
using Statistics
using LinearAlgebra
using Distributions
using Random
using RCall
using Plots

include("./samplers/fosr_dp.jl")
include("./samplers/fosr_dppm.jl")
include("./samplers/shared_samplers.jl")
include("./samplers/dp_samplers.jl")
include("./samplers/dppm_samplers.jl")
include("./samplers/helpers.jl")

Random.seed!(1234);

R"""
load("~/Research/fosr_clust/data/asfr.RData")
X = asfr_list$X
Y = asfr_list$Y
var_names = colnames(X)
"""

@rget X; @rget Y; @rget var_names;
N, T = size(Y);
M = 7;
n_iter = 10000;
burn_in = Int(n_iter / 10);
seq_to_keep = Array{Int64}(burn_in:n_iter);
W = Array{Float64, 2}(undef, N, 1);
W[:,1] = X[:,1];
X = X[:,2:end];
var_names = var_names[2:end];

# get relevant stuff for the model fits
time_eval = range(0, stop = 1, length = T) |> collect;
fourier_mat, mtheta = get_basis_mats(N, T, 3, M, time_eval);
diff_mat = get_diff_mat(2, M);
R = 0.001 * Diagonal{Float64}(I, M) + diff_mat' * diff_mat;

# dp
funcs_array_dp, d_mat_dp, tau_vec, eta_vec = fosr_dp(Y, W, X, mtheta, R, n_iter);
estim_d_ind_dp = best_d_ind(d_mat_dp[:, seq_to_keep]);
D_dp = 1 .- make_freq_mat(d_mat_dp[:,seq_to_keep])
dp_clust = get_clust_dict(estim_d_ind_dp, var_names, false) |> sort
estim_dp, lower_dp, upper_dp = get_estim_funcs(funcs_array_dp, seq_to_keep);

# dpmm
funcs_array_dppm, d_mat_dppm, tau_vec, eta_vec = fosr_dppm(Y, W, X, mtheta, R, n_iter);
estim_d_ind_dppm = best_d_ind(d_mat_dppm[:, seq_to_keep]);
D_dppm = 1 .- make_freq_mat(d_mat_dppm[:,seq_to_keep])
dppm_clust = get_clust_dict(estim_d_ind_dppm, var_names, false) |> sort
estim_dppm, lower_dppm, upper_dppm = get_estim_funcs(funcs_array_dppm, seq_to_keep);
d_probs = mean(d_mat_dppm[:,seq_to_keep] .== 0, dims = 2)

@rput D_dp; @rput var_names; @rput estim_dp; @rput lower_dp; @rput upper_dp;
@rput D_dppm; @rput var_names; @rput estim_dppm; @rput lower_dppm; 
@rput upper_dppm; @rput d_probs;

"""R

# setup for the plots
setwd("~/Research/fosr_clust/")
library(stats)
library(Hmisc)
library(corrplot)

xlabs = c("15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49")

# plot of asfr data for every country
pdf("asfr_plot.pdf", width = 14, height = 7)
par(mar = c(5, 4, 4, 2) + 1)
matplot(t(Y), type = "l", xaxt = "n", xlab = NA, 
        ylab = NA,
        main = "Age-specific Fertility Rates", cex.axis = 1.5,
        cex.main = 2)
xtick = seq(1, 7, by = 1)
axis(side = 1, at = xtick, labels = FALSE)
text(x = xtick, par("usr")[3], labels = xlabs, srt = 0, pos = 1, xpd = T, 
     offset = 1, cex = 1.5)
title(ylab = "Births per 1,000 Women", cex.lab = 2)
title(xlab = "Age Group", cex.lab = 2)
dev.off()

# table of probabilities of inclusion for the dppm model
df_probs = data.frame(var_names, d_probs) %>% arrange(d_probs)

prob_mat = df_probs$d_probs %>% as.matrix()
colnames(prob_mat) = "Percent Zero"
rownames(prob_mat) = c(
    "Contraception Prevalence (15-49)", 
    "Maternity Related Deaths per 1K Women", 
    "Births Atten. by Exp. Staff (% of Total)", 
    "Age at First Marriage", 
    "Male to Female Sex Ratio (15-49)",
    "U5 Mortality", 
    "Cervical Cancer Deaths (Per 100K)", 
    "Health Expenditures (% of GDP)",
    "Female BMI", 
    "Female Labor Force Participation", 
    "GDP Per Capita", 
    "Mean Yrs. of School (Women % Men) (25-34)", 
    "Life Expectancy", 
    "Dollar Billionares (per 1M)", 
    "Alcohol Consumption per Adult")

latex(prob_mat %>% round(3), file = "pred_tab.tex", table.env = FALSE, 
    title = "")

# column names for the dendograms
col_names = c(
    "Age at First Marriage", 
    "U5 Mortality", 
    "Contraception Prevalence (15-49)", 
    "Life Expectancy", 
    "GDP Per Capita", 
    "Health Expenditures (% of GDP)",
    "Female BMI", 
    "Dollar Billionares (per 1M)", 
    "Births Atten. by Exp. Staff (% of Total)", 
    "Mean Yrs. of School (Women % Men) (25-34)", 
    "Alcohol Consumption per Adult",
    "Cervical Cancer Deaths (Per 100K)", 
    "Female Labor Force Participation", 
    "Maternity Related Deaths per 1K Women", 
    "Male to Female Sex Ratio (15-49)")
    
# making cluster plot for DP model
hcc_dp = hclust(as.dist(D_dp))
postscript("dp_dend.eps", height = 16, width = 8)
par(cex = 1.75)
plot(hcc_dp, labels = col_names, main = "DP", xlab = "", sub = "")
dev.off()

# cluster plot for dppm model
hcc_dppm = hclust(as.dist(D_dppm))
pdf("dppm_dend.pdf", width = 8, height = 16)
par(cex = 1.75)
plot(hcc_dppm, labels = col_names, main = "DPPM", xlab = "", sub = "")
dev.off()

#**************************************************#
# making plots for the non-zero predictors
#**************************************************#

set_width = 11.5
set_height = 8

i = 2
this_var_name = "Age at First Marriage"
ylim = c(min(c(lower_dp[,i], lower_dppm[,i])), max(c(upper_dp[,i], upper_dppm[,i])))
pdf("age_fm.pdf", width = set_width, height = set_height)
plot(estim_dp[,i], type = "l", ylim = ylim, xlab = "", xaxt = "n", ylab = "", 
    main = this_var_name, cex.axis = 2, cex.main = 2)
lines(lower_dp[,i], type = "l", lty = 2)
lines(upper_dp[,i], type = "l", lty = 2)
lines(estim_dppm[,i], col = "red")
lines(lower_dppm[,i], col = "red", lty = 2)
lines(upper_dppm[,i], col = "red", lty = 2)
abline(h = 0, lty = 3, lwd = 0.5)
xtick = seq(1, 7, by = 1)
axis(side = 1, at = xtick, labels = FALSE)
text(x = xtick, par("usr")[3], labels = xlabs, srt = 0, pos = 1, xpd = T, 
    offset = 1, cex = 2)
legend(4.5, -15, legend = c("FOSR-DP", "FOSR-DPPM"), col = c("black", "red"), 
    lty = c(1, 1), cex = 2)
dev.off()

i = 3
this_var_name = "Under 5 Mortality"
ylim = c(-5, 28)
pdf("u5_mort.pdf", width = set_width, height = set_height)
plot(estim_dp[,i], type = "l", ylim = ylim, xlab = "", xaxt = "n", ylab = "", 
    main = this_var_name, cex.axis = 2, cex.main = 2)
lines(lower_dp[,i], type = "l", lty = 2)
lines(upper_dp[,i], type = "l", lty = 2)
lines(estim_dppm[,i], col = "red")
lines(lower_dppm[,i], col = "red", lty = 2)
lines(upper_dppm[,i], col = "red", lty = 2)
abline(h = 0, lty = 3, lwd = 0.5)
xtick = seq(1, 7, by = 1)
axis(side = 1, at = xtick, labels = FALSE)
text(x = xtick, par("usr")[3], labels = xlabs, srt = 0, pos = 1, xpd = T, 
    offset = 1, cex = 2)
legend(4.75, 25, legend = c("FOSR-DP", "FOSR-DPPM"), col = c("black", "red"), 
    lty = c(1, 1), cex = 2)
dev.off()

i = 15
this_var_name = "Maternal Deaths per 1-K"
ylim = c(-5, 28)
pdf("matern_deaths.pdf", width = set_width, height = set_height)
plot(estim_dp[,i], type = "l", ylim = ylim, xlab = "", xaxt = "n", ylab = "", 
    main = this_var_name, cex.axis = 2, cex.main = 2)
lines(lower_dp[,i], type = "l", lty = 2)
lines(upper_dp[,i], type = "l", lty = 2)
lines(estim_dppm[,i], col = "red")
lines(lower_dppm[,i], col = "red", lty = 2)
lines(upper_dppm[,i], col = "red", lty = 2)
abline(h = 0, lty = 3, lwd = 0.5)
xtick = seq(1, 7, by = 1)
axis(side = 1, at = xtick, labels = FALSE)
text(x = xtick, par("usr")[3], labels = xlabs, srt = 0, pos = 1, xpd = T, 
    offset = 1, cex = 2)
legend(4.75, 25, legend = c("FOSR-DP", "FOSR-DPPM"), col = c("black", "red"), 
    lty = c(1, 1), cex = 2)
dev.off()

i = 4
this_var_name = "Contraception Use"
ylim = c(min(c(lower_dp[,i], lower_dppm[,i])), max(c(upper_dp[,i], upper_dppm[,i])))
ylim = c(-27, 16)
pdf("contra_use.pdf", width = set_width, height = set_height)
plot(estim_dp[,i], type = "l", ylim = ylim, xlab = "", xaxt = "n", ylab = "", 
    main = this_var_name, cex.axis = 2, cex.main = 2)
lines(lower_dp[,i], type = "l", lty = 2)
lines(upper_dp[,i], type = "l", lty = 2)
lines(estim_dppm[,i], col = "red")
lines(lower_dppm[,i], col = "red", lty = 2)
lines(upper_dppm[,i], col = "red", lty = 2)
abline(h = 0, lty = 3, lwd = 0.5)
xtick = seq(1, 7, by = 1)
axis(side = 1, at = xtick, labels = FALSE)
text(x = xtick, par("usr")[3], labels = xlabs, srt = 0, pos = 1, xpd = T, 
    offset = 1, cex = 2)
legend(4, 12.25, legend = c("FOSR-DP", "FOSR-DPPM"), col = c("black", "red"), 
    lty = c(1, 1), cex = 2)
dev.off()

i = 10
this_var_name = "Births Attended by Experience Staff (%)"
ylim = c(min(c(lower_dp[,i], lower_dppm[,i])), max(c(upper_dp[,i], upper_dppm[,i])))
ylim = c(-27, 16)
pdf("birth_staff.pdf", width = set_width, height = set_height)
plot(estim_dp[,i], type = "l", ylim = ylim, xlab = "", xaxt = "n", ylab = "", 
    main = this_var_name, cex.axis = 2, cex.main = 2)
lines(lower_dp[,i], type = "l", lty = 2)
lines(upper_dp[,i], type = "l", lty = 2)
lines(estim_dppm[,i], col = "red")
lines(lower_dppm[,i], col = "red", lty = 2)
lines(upper_dppm[,i], col = "red", lty = 2)
abline(h = 0, lty = 3, lwd = 0.5)
xtick = seq(1, 7, by = 1)
axis(side = 1, at = xtick, labels = FALSE)
text(x = xtick, par("usr")[3], labels = xlabs, srt = 0, pos = 1, xpd = T, 
    offset = 1, cex = 2)
legend(4, 12.25, legend = c("FOSR-DP", "FOSR-DPPM"), col = c("black", "red"), 
    lty = c(1, 1), cex = 2)
dev.off()

i = 16
this_var_name = "Male to Female Sex Ratio (15-49)"
ylim = c(min(c(lower_dp[,i], lower_dppm[,i])), max(c(upper_dp[,i], upper_dppm[,i])))
pdf("sex_ratio.pdf",  width = set_width, height = set_height)
plot(estim_dp[,i], type = "l", ylim = ylim, xlab = "", xaxt = "n", ylab = "", 
    main = this_var_name, cex.axis = 2, cex.main = 2)
lines(lower_dp[,i], type = "l", lty = 2)
lines(upper_dp[,i], type = "l", lty = 2)
lines(estim_dppm[,i], col = "red")
lines(lower_dppm[,i], col = "red", lty = 2)
lines(upper_dppm[,i], col = "red", lty = 2)
abline(h = 0, lty = 3, lwd = 0.5)
xtick = seq(1, 7, by = 1)
axis(side = 1, at = xtick, labels = FALSE)
text(x = xtick, par("usr")[3], labels = xlabs, srt = 0, pos = 1, xpd = T, 
    offset = 1, cex = 2)
legend(3.5, -3, legend = c("FOSR-DP", "FOSR-DPPM"), col = c("black", "red"), 
    lty = c(1, 1), cex = 2)
dev.off()
"""

R"""

cols = colorRampPalette(rev(c("#67001F", "#B2182B", "#D6604D", "#F4A582",
                "#FDDBC7", "#FFFFFF", "#D1E5F0", "#92C5DE",
                "#4393C3", "#2166AC", "#053061")))
colnames(X)[15] = "matern_deaths"
tX = X[,-1]
corr_mat = cor(tX)
rownames(corr_mat) = col_names
colnames(corr_mat) = rep(" ", length(col_names))
pdf("corr_plot.pdf",  width = 16, height = 8)
corrplot(corr_mat, order = "hclust", tl.cex = 1.5, cl.cex = 1.5, tl.col = "black", 
    type = "lower", col = cols(50))
dev.off()

"""









