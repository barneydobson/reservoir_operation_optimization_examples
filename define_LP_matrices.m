function [f,A,b,LB,UB ] = define_LP_matrices(S_ini,I,d,S_cap,tol)
%
% [f,A,b,LB,UB ] = define_LP_matrices(S_ini,I,d,S_cap,R_cap,tol)
%
% Define the matrices f,A,b,LB,UB to reformulate the 
% 'simple supply reservoir' problem as a Linear Programming one.
%
% LP problem must be in the form:
%        min(f'*x)
% subject to A*x <= b
%            x<=UB
%            x>=LB

% Definition of the decision variables (x)

% We set:
% x = [u1,u2,u3,...,uT,w1,w2,w3,...wT]
% where u's are releases and w's are spills.
% Notice that spills must be treated as design variables 
% since they vary based on the choice of u.

% Definition of the objective function (vector f)

% we cannot use the original objective function because it is not linear in
% the decision variables.  
% However, we can use linear deficits:
%      [ (d1 - u1) + (d2 - u2) + ... + (dT - uT) ] 
% and prevent unnecessary spills by including them as additional costs in
% the linear objective. 
% Since the demand is a given constant, the coefficients in f relative to
% the u are all -1. As for the penalisation of the w, we use a set of
% coefficients all >= than the tolerance (tol) of the solver
% Hence:
T = length(I) ;
f = [
    -ones(T,1);      % Releases
    ((T:-1:1).*tol)' % Spills
    ];

% Definition of the constraints (A,b,LB,UB):

% First, we need to impose that storages respect the mass balance
% equations: 
%    St = St-1 + It - ut - wt
% 
% Specifically, starting from t=1 and iterating for each time step
% in the simulation period, we get:
%
% Timestep 1: u1 + w1           <= S_ini + I1
% Timestep 2: u2 + w2 + u1 + w2 <= S_ini + I1 + I2
% ...
% Timestep T: cumsum(u) + cumsum(w) <= S_ini + cumsum(I)
%
% To get this into the form A*x <= b, we must put use a lower triangular
% matrix for 'x' to replicate the cumsum behaviour for u and w.

lower_triangular = zeros(T);
for i = 1 : T
    lower_triangular(i,1:i) = 1;
end
A = [lower_triangular,lower_triangular]; % releases + spills up to a given
% time has to be less than initial storage plus inflows up to a given time
b = S_ini + cumsum(I);

% We also need to ensure that there is no oversupply, ie:
% ut <= dt
% 
% To get this into the form A*x <= we simply need the 'eye' function to
% create a diagonal of 1s that represents the releases
A = [A;[eye(T),zeros(T)]]; % releases <= demands
b = [b;d];

% Finally, spill must respect the equation:
% wt = max(0,S_cap - St)
%
% The constraint is equivalent to the following set of inequalities
% (provided spills w are to be minimized, which they are by their positive
% value in the objective function):

% Timestep 1: u1 + w1 >= S_ini + I1 - S_cap
% ...
% Timestep T: cumsum(u) + cumsum(w) >= S_ini + cumsum(I) - S_cap
%
% As with mass balance, we need to use the lower triangular matrix to
% replicate the cumsum behaviour for u and w. We also need to flip the
% signs since A*x <= b and not A*x >= b.
% Hence:
A = [A;[-lower_triangular,-lower_triangular]]; %releases + spills up to a given time must be more than initial storage plus inflows up to a given time minus storage cap. Since spills are to be minimized (positive number in objective function) - they will be used to make the difference but no more
b = [b;(S_cap - (S_ini + cumsum(I)))];

% Last, we define upper and lower bounds of u and w:
UB = [d;I];
LB = zeros(T*2,1);
