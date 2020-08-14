clear;
clc;
%% Load data
% Load river flow data:
flows = dlmread('example_data.txt'); % load time series of daily flow records
% Define simulation scenario:
year_start_inds = find(flows(:,1) == 1);
% Calibration period:
flows_1 = flows(1:year_start_inds(10) - 1,2)   ; %extract first 10 years (for example)
flows_1 = flows_1(1:end-mod(numel(flows_1),7)) ; %only include up to complete weeks
I_1 = sum(reshape(flows_1,7,numel(flows_1)/7))'; %aggregate flows to weekly (for speed)
% Test period:
flows_2 = flows(year_start_inds(11):(year_start_inds(20) - 1),2); %11th to 20th years (for example)
flows_2 = flows_2(1:end-mod(numel(flows_2),7)); %only include up to complete weeks
I_2     = sum(reshape(flows_2,7,numel(flows_2)/7))'; %aggregate flows to weekly (for speed)

T     = size(I_1,1)     ; % Length of simulation period (weeks)
d     = ones(T,1).*40*7 ; % Time series of (constant) water demand
S_ini = 500 ; % Initial storage
% Define other system parameters:
S_cap = 5000; % storage capacity

%Note: This is for the purpose of demonstrating operation optimization
%techniques only. It is not an attempt at a rigorous experiment to
%differentiate between the performance of different optimization
%techniques, and should not be interpreted as so.

%% Optimize a RS against inflows
%
% To revise example1_RS_optimization.m let us optimize RS inflows for our
% test period (I_2). This will give usthe maximum attainable performance
% possible:
tol   = 10e-6 ; % 
[f,H,A,b,LB,UB ] = define_QP_matrices(S_ini,I_2,d,S_cap,tol);
xopt_QP_2  = quadprog(H,f',A,b,[],[],LB,UB,[]);
opt_u_QP_2 = xopt_QP_2(1:T);
opt_w_QP_2 = xopt_QP_2(T+1:end);
opt_S_QP_2 = S_ini + cumsum(I_2) - cumsum(opt_u_QP_2) - cumsum(opt_w_QP_2);
opt_J_QP_2 = sum( ( d - opt_u_QP_2 ).^2 ) ;
figure(1)
subplot(4,1,1)
plot(I_2)       ; hold on; xlabel('time (weeks)');ylabel('inflow (Ml/week)');
subplot(4,1,2)
plot(opt_u_QP_2)    ; hold on; xlabel('time (weeks)');ylabel('release (Ml/week)');
subplot(4,1,3)
plot(d - opt_u_QP_2); hold on; xlabel('time (weeks)');ylabel('deficit (Ml/week)');
legend([ 'opt QP (J= ' num2str(opt_J_QP_2) ')'])
subplot(4,1,4)
plot(opt_S_QP_2)    ; hold on; xlabel('time (weeks)');ylabel('storage (Ml)');

%% Applying a RS against inflows it was not optimised for
%
% First we must optimize relases to the calibration period, for example
% using QP (this is just as in previous example):
tol   = 10e-6 ; % 
[f,H,A,b,LB,UB ] = define_QP_matrices(S_ini,I_1,d,S_cap,tol);
xopt_QP  = quadprog(H,f',A,b,[],[],LB,UB,[]);
u_QP_1 = xopt_QP(1:T);
w_QP_1 = xopt_QP(T+1:end);
S_QP_1 = S_ini + cumsum(I_1) - cumsum(u_QP_1) - cumsum(w_QP_1);
J_QP_1 = sum( ( d - u_QP_1 ).^2 ) ;

% We can now simulate these releases against a different set of flows
% (flows_2) using the 'simulate_RS' function:
[J_QP_2,S_QP_2,u_QP_2,w_QP_2] = simulate_RS(u_QP_1,d,I_2,S_cap,S_ini);

figure(1)
subplot(4,1,2)
plot(u_QP_2)    ; hold on; xlabel('time (weeks)');ylabel('release (Ml/week)');
subplot(4,1,3)
plot(d - u_QP_2); hold on; xlabel('time (weeks)');ylabel('deficit (Ml/week)');
legend([ 'QP (J= ' num2str(J_QP_2) ')'])
subplot(4,1,4)
plot(S_QP_2)    ; hold on; xlabel('time (weeks)');ylabel('storage (Ml)');
%We see that the just implementing an optimized release sequence over a
%set of flows that it wasn't optimized for performs very badly.

%% Formulating an operating policy
%
% We will formulate a simple operating policy that is linearly dependent on
% storage and inflows:
%
% u(t) = theta(1) + theta(2)*S(t) + theta(3)*I(t)

%% Standard Operating Policy
% The standard operating policy is a commonly assumed 'baseline' operation
% scenario. It is simply to satisfy demand if possible, and otherwise
% satisfy demand as much as is possible (i.e. releasing all available
% storage).
%
% What we have described is to simply set u(t) = d(t), since our 
% simulate_OP resolves constraints, this will by definition implement the 
% standard operating policy
[J_SOP_2,S_SOP_2,u_SOP_2,w_SOP_2] = simulate_OP([d(1),0,0],d,I_2,S_cap,S_ini);
figure(1)
subplot(4,1,2)
plot(u_SOP_2)
subplot(4,1,3)
plot(d - u_SOP_2)
legend([ 'opt QP (J= ' num2str(opt_J_QP_2) ')'],[ 'QP (J= ' num2str(J_QP_2) ')'],[ 'SOP (J= ' num2str(J_SOP_2) ')'])
subplot(4,1,4)
plot(S_SOP_2)
%The SOP appears to be marginally better than using the set of flows
%optimized to the calibration set over the validation period. But, having
%just read a review on reservoir operation optimization, we can probably do
%a bit better!

%% Release Sequence Based (RSB) optimisation of the OP
%
% One way to optimise the OP is to infer its optimal parameters by
% regression of the time series of optimal storages-releases obtained in
% the calibration period (1):
rsb_training_data = [ones(T-1,1), S_QP_1(1:(T-1)), I_1(2:T)] ;
rsb_parameters    = regress(u_QP_1(2:T),rsb_training_data)   ;
% We can now simulate the system over the test period (2) using the RSB
% operating policy:
[J_RSB_2,S_RSB_2,u_RSB_2,w_RSB_2] = simulate_OP(rsb_parameters,d,I_2,S_cap,S_ini);

figure(1)
subplot(4,1,2)
plot(u_RSB_2)
subplot(4,1,3)
plot(d - u_RSB_2)
legend([ 'opt QP (J= ' num2str(opt_J_QP_2) ')'],[ 'QP (J= ' num2str(J_QP_2) ')'],[ 'SOP (J= ' num2str(J_SOP_2) ')'],[ 'RSB (J= ' num2str(J_RSB_2) ')'])
subplot(4,1,4)
plot(S_RSB_2)
%We see that the RSB approach isn't as good as the optimal set of releases
%over the validation set of flows, but it is clearly performing much better
%than using the set of flows optimized to the calibration set.

%% Direct Policy Search
%
% Another possible approach to optimise the OP is to directly infer the
% parameters that optimise the objective function over the calibration
% period (1). This can be achieved for example using a GA:
rng(1000000); % set seed
dps_parameter = ga(@(x) simulate_OP(x,d,I_1,S_cap,S_ini),3);
% We can now simulate the system over the test period (2) using the DPS
% operating policy:
[J_DPS_2,S_DPS_2,u_DPS_2,w_DPS_2] = simulate_OP(dps_parameter,d,I_2,S_cap,S_ini);
figure(1)
subplot(4,1,2)
plot(u_DPS_2)
subplot(4,1,3)
plot(d - u_DPS_2)
legend([ 'opt QP (J= ' num2str(opt_J_QP_2) ')'],[ 'QP (J= ' num2str(J_QP_2) ')'],[ 'SOP (J= ' num2str(J_SOP_2) ')'],[ 'RSB (J= ' num2str(J_RSB_2) ')'],[ 'DPS (J= ' num2str(J_DPS_2) ')'])
subplot(4,1,4)
plot(S_DPS_2)
%We also see that the DPS approach seems to improve the policy in general
%(although this depends on the seeds - try a few to get an idea).
