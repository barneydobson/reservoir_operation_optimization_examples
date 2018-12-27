clear;clc;
%% Load data
storage_cap = 5000; %storage capacity
flows = dlmread('example_data.txt'); %load large set of flows
flows = flows(1:364*1,2); %extract first year (for example)
flows = sum(reshape(flows,7,numel(flows)/7))'; %aggregate flows to weekly (for speed)
T = size(flows,1);
storage_initial = 500;
demand = ones(T,1).*40*7; %Daily demand 40, weekly 40*7
tol = 10e-6;
release_cap = 40*7;


%% Linear programming
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Takes the form min(f'*x), subject to A*x <= b, where x are design 
% variables
% 
% Design variables:
% [u1,u2,u3,...,uT,w1,w2,w3,...wT], u's are releases and w's are spills
% n.b. spills must be treated as design variables since they vary based on 
% the choice of u.
% 
% Objective:
% -Deficit, [d1 - u1 + d2 - u2 + ... + dT - uT], d's are demands
% -Since demands are constants we simply need to minimize -u
% -To prevent unnecessary spill we must penalise it by a value greater than
% the tolerance of the solver
% 
% Constraints:
% -Mass balance, St = St-1 + It - ut - wt, S is storage and I's are inflows
% -St varies based on the choice of ut, however we can formulate this
% without having to treat it as a design variable as we did with wt
%
% -Timestep 1: u1 + w1 <= S0 + I1, S0 is storage_initial
% -Timestep 2: u2 + w2 + u1 + w2 <= S0 + I1 + I2
% ...
% -Timestep T: cumsum(u) + cumsum(w) <= S0 + cumsum(I)
%
% -To get this into the form A*x <= b, we must put use a lower triangular
% matrix for 'x' to replicate the cumsum behaviour for u and w.
%
% No oversupply:
% -ut <= dt
% -To get this into the form A*x <= we simply need the 'eye' function to
% create a diagonal of ones that represents the releases
% 
% %Spill:
% wt = max(0,Scap - St), Scap is storage_cap
% -Again, S varies based on choice of u, so we formulate this
% differently. The following works provided spills are to be minimized.
% (which they are by their positive value in the objective function).
% -Timestep 1: u1 + w1 >= S0 + I1 - Scap
% ...
% -Timestep T: cumsum(u) + cumsum(w) >= S0 + cumsum(I) - Scap
% -As with mass balance, we need to use the lower triangular matrix to
% replicate the cumsum behaviour for u and w. We also need to flip the
% signs since A*x <= b and not A*x >= b.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Objective function
f = [
    -ones(T,1); %Decisions
    ((T:-1:1).*tol)' %Spills
    ];

%Constraints

%Mass balance
lower_triangular = zeros(T);
for i = 1 : T
    lower_triangular(i,1:i) = 1;
end
A = [lower_triangular,lower_triangular]; %releases + spills up to a given time has to be less than initial storage plus flows up to a given time
b = storage_initial + cumsum(flows);
%No oversupply
A = [A;[eye(T),zeros(T)]]; %releases <= demands
b = [b;demand];
%Spill
A = [A;[-lower_triangular,-lower_triangular]]; %releases + spills up to a given time must be more than initial storage plus flows up to a given time minus storage cap. Since spills are to be minimized (positive number in objective function) - they will be used to make the difference but no more
b = [b;(storage_cap - (storage_initial + cumsum(flows)))];
%Upper and lower bounds
UB = [ones(T,1).*release_cap;flows];
LB = zeros(T*2,1);

%Run optimization
xopt_lp = linprog(f',A,b,[],[],LB,UB,[]);
releases_lp = xopt_lp(1:T);
spills_lp = xopt_lp(T+1:end);
%Following the mass balance present above we can create storage
storage_lp = storage_initial + cumsum(flows) - cumsum(releases_lp) - cumsum(spills_lp);
%Plot results
subplot(4,1,1);
plot(flows); hold on;xlabel('time (weeks)');ylabel('inflow (Ml/week)');
subplot(4,1,2);
plot(releases_lp); hold on;xlabel('time (weeks)');ylabel('release (Ml/week)');
subplot(4,1,3);
plot(demand - releases_lp); hold on;xlabel('time (weeks)');ylabel('deficit (Ml/week)');
subplot(4,1,4);
plot(storage_lp); hold on;xlabel('time (weeks)');ylabel('storage (Ml)');
%We see that the start of the timeseries has very low flows and so the
%storage gets drawn down causing large deficit. Although the flows during 
%weeks 8 to 20 replenish the storage, they are not sufficient and again a
%large deficit occurs during weeks 20-40. After week 40 the flows are high
%and the storage can restore.

%% Quadratic programming
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Takes the form min(0.5*x'*H*x + f'*x), subject to A*x <= b
% 
% Design variables and constraints are the same as linear programming
%
% Objective is now:
% Deficit, [(d1 - u1)^2 + (d2 - u2)^2 + ... + (dT - uT)^2]
%  = [d1^2 + d2^2 + .. dT^2] + [u1^2 + u2^2 + .. + uT^2] - [2*d1*u1 +
%  2*d2*u2 + .. + 2*dT*uT]
% -The d^2 part of this is constant and so can be omitted.
% -The u^2 part of this can be accounted by putting 4's in the H matrix.
% Because it is a square matrix we must use the eye function.
% -The LP objective was -u, the linear part of this objective is 2*d*u,
% thus we can just use the LP objective by multiplying it by 4*d.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create new objective
f(1:T) = f(1:T)*4.*demand;
H = [2.*eye(T),zeros(T);zeros(T,T*2)];
%Run optimization
xopt_qp = quadprog(H,f',A,b,[],[],LB,UB,[]);
releases_qp = xopt_qp(1:T);
spills_qp = xopt_qp(T+1:end);
storage_qp = storage_initial + cumsum(flows) - cumsum(releases_qp) - cumsum(spills_qp);
%Plot results
subplot(4,1,2);
plot(releases_qp); 
subplot(4,1,3);
plot(demand - releases_qp); 
subplot(4,1,4);
plot(storage_qp);
%Instead we see a completely different picture. In an attempt to minimize
%the squared deficits the deficit is 'spread out' over a long period. This
%causes more days of deficit, but a smaller maximum deficit.

%% GA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We could use a GA to solve the exact same problem as QP, (you can try
% this with ga(@(x) 0.5*x*H*x' + f'*x',numel(xopt_qp),A,b,[],[],LB,UB).
% However QP is efficient because of it's ability to exploit constraints,
% something that GA's do not do. Instead we should use a simulation
% approach that imposes constraints 'on the fly', so that the GA doesn't 
% have to waste time finding the feasible region. We will use the function
% 'simulate_releases' - see the .m file.
%
% The optimization also outputs 'target_releases_ga' instead of 
% 'releases_ga' because the constraints are applied within the simulation
% function, thus the output from the GA is not necessarily feasible (and so
% we must simulate the target releases to determine the actual releases).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
target_releases_ga = ga(@(x) simulate_releases(x,demand,flows,storage_cap,storage_initial),T)';
[~,storage_ga,releases_ga,spills_ga] = simulate_releases(target_releases_ga,demand,flows,storage_cap,storage_initial);
%Plot results
subplot(4,1,2);
plot(releases_ga); legend('LP','QP','GA');
subplot(4,1,3);
plot(demand' - releases_ga); legend('LP','QP','GA');
subplot(4,1,4);
plot(storage_ga);legend('LP','QP','GA');

%We see that the GA produces similar results to QP (which we would expect
%because the simulate_releases.m function uses a quadratic objective). But
%what happens if you use a longer time series? Try solving a release 
%sequence for 10 years of data instead of 1 (change line 5 to: 
%flows = flows(1:364*10,2)). 
%
%The GA now performs quite a bit worse than the QP. This is because
%optimization gets harder the more variables there are. QP is very
%efficient at using constraints to navigate these large problem spaces, but
%the GA does not get these benefits.
