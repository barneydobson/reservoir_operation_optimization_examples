clear;clc;
%% Load data
% Load river flow data:
flows = dlmread('example_data.txt');  % load time series of daily flow records

% Define simualtion scenario:
flows = flows(1:364*10,2); % extract first year (for example)
I     = sum(reshape(flows,7,numel(flows)/7))'; % aggregate flows to weekly (for speed)
T     = size(I,1)       ; % Length of simulation period (weeks)
d     = ones(T,1).*40*7 ; % Time series of (constant) water demand
S_ini = 500 ; % Initial storage

% Define other system parameters
S_cap = 5000; %storage capacity


%% Linear programming

% LP can only solve problems in the form:
%        min(f'*x)
% subject to A*x <= b
%            x<=UB
%            x>=LB
% where x are the decision variables.
% In order to apply LP, we then need to reformulate our original
% reservoir optimisation problem in such a way that it fits the LP
% form above. This is possible with some level of approximation,
% given that the actual problem is not linear (for example the objective
% function uses squared deficits). The process of defining 
% <f,A,b,LB,UB> for our problem is set out in the function
% 'define_LP_matrices.m' (see the code and comments in there for more info)
tol   = 10e-6 ; % 
R_cap = d(1)  ; % maximum allowed release outflow
[f,A,b,LB,UB ] = define_LP_matrices(S_ini,I,d,S_cap,R_cap,tol) ;

% Run optimization using Matlab LP solver:
xopt_LP     = linprog(f',A,b,[],[],LB,UB,[]);
% Extract time series of release and spills:
u_LP = xopt_LP(1:T)       ; % optimal Release Sequence
w_LP   = xopt_LP(T+1:end) ; % sequence of spills
% Use mass balance equation to derive the time series of storage:
S_LP = S_ini + cumsum(I) - cumsum(u_LP) - cumsum(w_LP);
% Compute objective value associated to the LP solution:
J_LP = sum( ( d - u_LP ).^2 ) ;

% Plot results:
subplot(4,1,1);
plot(I)       ; hold on; xlabel('time (weeks)'); ylabel('inflow (Ml/week)');
subplot(4,1,2);
plot(u_LP)    ; hold on; xlabel('time (weeks)'); ylabel('release (Ml/week)');
subplot(4,1,3);
plot(d - u_LP); hold on; xlabel('time (weeks)'); ylabel('deficit (Ml/week)');
legend([ 'LP (J= ' num2str(J_LP) ')'])
subplot(4,1,4);
plot(S_LP)    ; hold on; xlabel('time (weeks)'); ylabel('storage (Ml)');
% We see that the start of the timeseries has very low flows and so the
% storage gets drawn down causing large deficit. Although the inflows during 
% weeks 8 to 20 replenish the storage, they are not sufficient and again a
% large deficit occurs during weeks 20-40. After week 40 the flows are high
% and the storage can restore.

%% Quadratic programming

% QP can only solve problems in the form:
%        min(0.5*x'*H*x + f'*x)
% subject to A*x <= b
%            x<=UB
%            x>=LB
% The process of defining <f,A,b,LB,UB> for our problem is set out in the
% function 'define_QP_matrices' (see the code and comments in there for
% more info). Notice that in this case, given that the actual problem has a
% quadratic objective function, the QP formulation is actually consistent
% with the original problem formulation.
tol   = 10e-6 ; %
R_cap = d(1)  ; % maximum allowed release outflow
[f,H,A,b,LB,UB ] = define_QP_matrices(S_ini,I,d,S_cap,R_cap,tol);

% Run optimization using Matlab QP solver:
xopt_QP = quadprog(H,f',A,b,[],[],LB,UB,[]);
% Extract time series of release and spills:
u_QP  = xopt_QP(1:T)    ; % optimal Release Sequence
w_QP  = xopt_QP(T+1:end); % sequence of spills
% Use mass balance equation to derive time series of storage:
S_QP  = S_ini + cumsum(I) - cumsum(u_QP) - cumsum(w_QP);
% Compute objective value associated to the QP solution:
J_QP = sum( ( d - u_QP ).^2 ) ;

% Plot results:
subplot(4,1,2);
plot(u_QP); 
subplot(4,1,3);
plot(d - u_QP); 
legend([ 'LP (J= ' num2str(J_LP) ')'],[ 'QP (J= ' num2str(J_QP) ')']);
subplot(4,1,4);
plot(S_QP);
% QP solution behaves quite differently from LP solution.
% In an attempt to minimize the squared deficits, QP reduces releases all
% along the simulation period, so that deficits are 'spread out' over a
% long period. This causes more days of deficit, but a smaller maximum
% deficit. The optimal objective function for QP is thus lower than LP.

%% GA

% We could use a GA to solve the exact same problem as QP, (you can try
% this with ga(@(x) 0.5*x*H*x' + f'*x',numel(xopt_qp),A,b,[],[],LB,UB).
% However QP is efficient because of its ability to exploit the linear nature
% of the constraints set, something that GAs do not do. 
% Instead we should use a simulation approach that imposes constraints 'on
% the fly', so that the GA does not waste time in finding the feasible
% region. We will use the function 'simulate_releases' to implement
% these constraints, and compute the value of the objective function for a
% given choice of the release sequence (u).
% We then use the Matlab GA function to optimise the release sequence:
u_GA = ga(@(u) simulate_RS(u,d,I,S_cap,S_ini),T)';

% Before we look at the optimised release sequence produced by 'ga', we
% need to apply the 'simulate_RS' function one last time in order to ensure
% that the u_GA is actually feasible (in fact, the output from the 'ga' is
% not necessarily feasible, given that the feasibility checks are performed
% by 'simulate_RS'):
[J_GA,S_GA,u_GA,w_GA] = simulate_RS(u_GA,d,I,S_cap,S_ini);

% Plot results
subplot(4,1,2);
plot(u_GA,'o'); legend('LP','QP','GA');
subplot(4,1,3);
plot(d - u_GA,'o'); legend('LP','QP','GA');
legend([ 'LP (J= ' num2str(J_LP) ')'],[ 'QP (J= ' num2str(J_QP) ')'],[ 'GA (J= ' num2str(J_GA) ')']);
subplot(4,1,4);
plot(S_GA,'o');legend('LP','QP','GA');

% We see that the GA produces similar results to QP (which we would expect
% because the simulate_releases.m function uses a quadratic objective). 

% But what happens if you use a longer time series? Try solving a release 
% sequence for 10 years of data instead of 1 (change line above to: 
% flows = flows(1:364*10,2)). The GA now performs quite a bit worse than the QP. This is because
% optimization gets harder the more variables there are. QP is very
% efficient at using constraints to navigate these large problem spaces, but
% the GA does not get these benefits.



