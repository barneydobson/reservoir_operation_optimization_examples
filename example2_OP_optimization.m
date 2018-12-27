close all;clear;clc;
%% Load data
storage_cap = 5000; %storage capacity
flows= dlmread('example_data.txt'); %load large set of flows
year_start_inds = find(flows(:,1) == 1);
storage_initial = 500;
tol = 10e-6;
release_cap = 40*7;

%Calibration flows
flows_cal = flows(1:year_start_inds(10) - 1,2); %extract first 10 years (for example)
flows_cal = flows_cal(1:end-mod(numel(flows_cal),7)); %only include up to complete weeks
flows_cal = sum(reshape(flows_cal,7,numel(flows_cal)/7))'; %aggregate flows to weekly (for speed)

T = size(flows_cal,1);

%Validation flows
flows_val = flows(year_start_inds(11):(year_start_inds(20) - 1),2); %11th to 20th years (for example)
flows_val = flows_val(1:end-mod(numel(flows_val),7)); %only include up to complete weeks
flows_val = sum(reshape(flows_val,7,numel(flows_val)/7))'; %aggregate flows to weekly (for speed)


demand = ones(T,1).*40*7; %Daily demand 40, weekly 40*7

%Note: This is for the purpose of demonstrating operation optimization
%techniques only. It is not an attempt at a rigorous experiment to
%differentiate between the performance of different optimization
%techniques, and should not be interpreted as so.

%% Operating a reservoir system without 'perfect foresight'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%In example1_RS_optimization.m we optimized releases with 'perfect
%foresight' of future inflows. Of course in reality this is rarely, if
%ever, the case. What we would really like to know, is how does an optimal
%RS perform on inflows that it was not optimized on?
%
%First we must optimize relases to the calibration period, below we perform
%a condensed version of what is perfomed in example1_RS_optimization.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = [-4*demand;((T:-1:1).*tol)'];
H = [2.*eye(T),zeros(T);zeros(T,T*2)];
lower_triangular = zeros(T);
for i = 1 : T
    lower_triangular(i,1:i) = 1;
end
A = [lower_triangular,lower_triangular;...
    eye(T),zeros(T);...
    -lower_triangular,-lower_triangular]; 
b = [storage_initial + cumsum(flows_cal);...
    demand;...
    (storage_cap - (storage_initial + cumsum(flows_cal)))];
UB = [ones(T,1).*release_cap;flows_cal];
LB = zeros(T*2,1);
xopt_qp = quadprog(H,f',A,b,[],[],LB,UB,[]);
releases_qp_cal = xopt_qp(1:T);
spills_qp_cal = xopt_qp(T+1:end);
storage_qp_cal = storage_initial + cumsum(flows_cal) - cumsum(releases_qp_cal) - cumsum(spills_qp_cal);
[obj_qp_cal,~,~,~] = simulate_releases(releases_qp_cal,demand,flows_cal,storage_cap,storage_initial);

disp(['Using QP to optimize over calibration period gives performance = ' num2str(obj_qp_cal)]);

%For comparison lets optimize over the validation period too
UB = [ones(T,1).*release_cap;flows_val];
b = [storage_initial + cumsum(flows_val);...
    demand;...
    (storage_cap - (storage_initial + cumsum(flows_val)))];
xopt_qp_val = quadprog(H,f',A,b,[],[],LB,UB,[]);
[optimized_validation_performance,~,~,~] = simulate_releases(xopt_qp_val(1:T),demand,flows_val,storage_cap,storage_initial);
disp(['Using QP to optimize over validation period gives performance = ' num2str(optimized_validation_performance)]);
%This doesn't show us much, just that the validation period is probably a
%bit drier than the calibration period.

%We have used QP to determine the optimal set of releases for the
%two sets of flows (flows_cal/val). We can now simulate the releases
%optimized to the calibration set of flows (flows_cal) against the 
%validation set of flows (flows_val) using the 'simulate_releases' function 
%from example1_RS_optimization.m.
[obj_qp_val,storage_qp_val,releases_qp_val,spills_qp_val] = simulate_releases(releases_qp_cal,demand,flows_val,storage_cap,storage_initial);
disp(['Using QP releases optimized over calibration period to operate over validation period gives performance = ' num2str(obj_qp_val)]);
subplot(4,1,1);
plot(flows_val);hold on;xlabel('time (weeks)');ylabel('inflow (Ml/week)');
subplot(4,1,2);
plot(releases_qp_val);hold on;xlabel('time (weeks)');ylabel('release (Ml/week)');
subplot(4,1,3);
plot(demand' - releases_qp_val); hold on;xlabel('time (weeks)');ylabel('deficit (Ml/week)');
subplot(4,1,4);
plot(storage_qp_val);hold on;xlabel('time (weeks)');ylabel('storage (Ml)');
%We see that the just implementing an optimized release sequence over a
%set of flows that it wasn't optimized for performs very badly.

%% Formulating an operating policy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%It is clear that an operating policy approach is required.
%We will formulate a simple operating policy that is linearly dependent on
%storage and flows. Following the rule:
%
%release(t) = parameter(1) + parameter(2)*storage(t-1) + parameter(3)*flows(t)
%
%We provide a function 'simulate_policy.m' to simulate this policy.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Standard Operating Policy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The standard operating policy is a commonly assumed 'baseline' operation
% scenario. It is simply to satisfy demand if possible, and otherwise
% satisfy demand as much as is possible (i.e. releasing all available
% storage).
%
% What we have described is to simply set releases(t) = demand(t), since
% our simulate_policy resolves constraints, this will by definition
% implement the standard operating policy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[obj_sop_val,storage_sop_val,releases_sop_val,spills_sop_val] = simulate_policy([demand(1),0,0],demand,flows_val,storage_cap,storage_initial);
disp(['Using SOP to operate gives performance = ' num2str(obj_sop_val)]);
subplot(4,1,2);
plot(releases_sop_val);
subplot(4,1,3);
plot(demand - releases_sop_val);
subplot(4,1,4);
plot(storage_sop_val);
%The SOP appears to be marginally better than using the set of flows
%optimized to the calibration set over the validation period. But, having
%just read a review on reservoir operation optimization, we can probably do
%a bit better!

%% Release Sequence Based
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To start with we will try to fit a basic regression to the optimized release
% sequence from above. Fitting an operating policy to an optimized RS is the 
% method for 'Release Sequence Based' techniques.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rsb_training_data = [ones(T-1,1), storage_qp_cal(1:(T-1)), flows_cal(2:T)];
rsb_parameters = regress(releases_qp_cal(2:T),rsb_training_data);
[obj_rsb_val,storage_rsb_val,releases_rsb_val,spills_rsb_val] = simulate_policy(rsb_parameters,demand,flows_val,storage_cap,storage_initial);
disp(['Using RSB to operate gives performance = ' num2str(obj_rsb_val)]);
subplot(4,1,2);
plot(releases_rsb_val);
subplot(4,1,3);
plot(demand - releases_rsb_val);
subplot(4,1,4);
plot(storage_rsb_val);
%We see that the RSB approach isn't as good as the optimal set of releases
%over the validation set of flows, but it is clearly performing much better
%than using the set of flows optimized to the calibration set.

%% Direct Policy Search
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In contrast 'Direct Policy Search' directly optimizes the parameters of the
% operating policy. We use matlab's in built GA function (requires the matlab
% optimization toolbox) to repeatedly simulate operating policies to determine
% an approximately optimal operating policy.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(1000000);
ga_parameter = ga(@(x) simulate_policy(x,demand,flows_cal,storage_cap,storage_initial),3);
[obj_ga_val,storage_ga_val,releases_ga_val,spills_ga_val] = simulate_policy(ga_parameter,demand,flows_val,storage_cap,storage_initial);
disp(['Using GA to operate gives performance = ' num2str(obj_ga_val)]);
subplot(4,1,2);
plot(releases_ga_val);legend('QP for operation','SOP','RSB','DPS');
subplot(4,1,3);
plot(demand - releases_ga_val);legend('QP for operation','SOP','RSB','DPS');
subplot(4,1,4);
plot(storage_ga_val);legend('QP for operation','SOP','RSB','DPS');
%We also see that the DPS approach seems to improve the policy in general
%(although this depends on the seed, and you can experiment with different
%validation/calibration periods to see how consistent this is).
