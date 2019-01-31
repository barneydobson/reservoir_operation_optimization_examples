function [J,varargout] = simulate_OP(theta,d,I,S_cap,S_ini)
%
% [J,S,u,w] = simulate_OP(theta,d,I,S_cap,S_ini)

T = length(I);

w = nan(T,1);
u = nan(T,1);
S = nan(T+1,1); S(1) = S_ini;
for t = 1 : T
    % calculate release decision from the Operating Policy:
    u(t) = theta(1) + S(t)*theta(2) + I(t)*theta(3);
    % check that the release is feasible:
    u(t) = min(d(t),u(t)) ; % release must not exceed the demand
    u(t) = max(u(t),0)    ; % release must be >=0
    u(t) = min(u(t),S(t)+I(t) ); % release cannot exceed the available water volume
    % calculate next timestep storage by mass balance equation:
    S(t+1) = S(t) + I(t) - u(t);
    w(t)   = S(t+1) - min(S_cap,S(t+1));
    S(t+1) = S(t+1) - w(t);
end

J = sum((d - u).^2);
if nargout > 1
    varargout = {S(2:end), u, w};
end

