function [ H , idx_u ] = solve_VF( H_ , s , I , sys_param )
%
% [ H , idx_u ] = solve_VF( H_ , s , I , sys_param )
%
% Solve the deterministic Bellman equation
%
% H_min(s) = min( g(s,u,e) + H_(s,u,e) )
%             u
%
% for one reservoir with one inflow and one release decision
%
% Input:
% H_ = optimal cost-to-go function at forward time-step
%      (vector (n_s,1))
%  s = current storage value (scalar)
%  I  = deterministic inflow value (scalar)
%  sys_param = required parameters (matlab struct)
%
% Output:
%     H = optimal cost-to-go value at current time-step
%         (scalar)
% u_idx = index of optimal release decision (scalar/vector)
%
% Comment:
% "sys_param" must include the following fields:
% .discr_s - (n_s,1)
% .discr_u - (n_s,1)
% .S_cap   - scalar (reservoir capacity)
% .d       - scalar (water demand)

discr_s = sys_param.discr_s ;
discr_u = sys_param.discr_u ;
d = sys_param.d ;

% ::::::::::::::::: %
% compute release   %
% ::::::::::::::::: %
%
% The result of this block must be a vector "R"
% of size (n_u,1). The i-th element of "R" is the
% reservoir release in correspondence to the i-th
% decision value in "discr_u" and storage "s" and
% inflow "I" 

R = discr_u ; % ( n_u , 1 )
R = min( R, d )  ; % impose constraint u(t) <= d(t)
R = min( R, s + I ) ; % impose constraint u(t) <= s(t) + I(t) 

% ::::::::::::::::: %
% compute step-cost %
% ::::::::::::::::: %
%
% The result of this block must be a vector "G"
% of size (n_u,1). The i-th element of "G" is the
% step-cost in correspondence to the i-th
% decision value in "discr_u" and storage "s" and
% inflow "I" 

G = ( sys_param.d - R ).^2 ; % ( n_u , 1 )

% :::::::::::::::::: %
% compute cost-to-go %
% :::::::::::::::::: %
%
% The result of this block must be a vector "H_"
% of size (n_u,1). The i-th element of "H_" is the
% optimal cost-to-go in correspondence to the i-th
% decision value in "discr_u" and storage "s" and
% inflow "I" 

S_     = s + ( I - R )                         ; % ( n_u , 1 )
W      = max( 0, S_ - sys_param.S_cap ) ; % ( n_u , 1 )
S_     = S_ - W ; % impose spills so that storage never exceeds maximum capacity
H_     = interp1( discr_s , H_ , S_ )          ; % ( n_u , 1 )

% :::::::::::::::::::::: %
% solve Bellman equation %
% :::::::::::::::::::::: %
%
% This block solve the Bellman equation.
% It should not change from one case study to another.

Q     = G + H_                         ; % ( n_u , 1 )
sens  = 10^-10                          ;
H     = min(Q)                         ;
if H >0
    idx_u = find( abs(Q-H)/H <= sens ) ;
else
    idx_u = find( abs(Q-H) <= sens )   ;
end
