function [f,H,A,b,LB,UB ] = define_QP_matrices(S_ini,I,d,S_cap,tol)
%
% [f,H,A,b,LB,UB ] = define_QP_matrices(S_ini,I,d,S_cap,R_cap,tol)
%
% Define the matrices f,H,A,b,LB,UB to reformulate the 
% 'simple supply reservoir' problem as a Quadratic Programming one.
%
% QP problem must be in the form:
%        min(0.5*x'*H*x + f'*x)
% subject to A*x <= b
%            x<=UB
%            x>=LB

% Definition of the decision variables (x)

% We set:
% x = [u1,u2,u3,...,uT,w1,w2,w3,...wT]
% where u's are releases and w's are spills.
% Notice that spills must be treated as design variables 
% since they vary based on the choice of u.

% Definition of the objective function (vector f,H)
%
% Our original objective functions is the sum of the elements of the
% squared deficits vector:
%    [(d1 - u1)^2 + (d2 - u2)^2 + ... + (dT - uT)^2] 
% which we can rewrite as:
%    [d1^2 + d2^2 + .. dT^2] + [u1^2 + u2^2 + .. + uT^2] - 
%       + [2*d1*u1 + 2*d2*u2 + .. + 2*dT*uT]
% Now:
% - The d^2 part of the above vector is constant and so can be omitted (it
% does not affect minimisation)
% - The u^2 part can be accounted by putting 2's in the H matrix.
% - The 2*d*u is just releases multiplied by -2*d in the f matrix
% - To prevent unnecessary spill we add to our objective function 
% a penalisation of the w (by a value greater than the tolerance of the
% solver )
T = length(I) ;
f = [
    - 2.*d;      % Releases
    ((T:-1:1).*tol)' % Spills
    ];
H = [2.*eye(T),zeros(T);zeros(T,T*2)];

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
