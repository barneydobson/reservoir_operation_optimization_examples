function [J,varargout] = simulate_RS(u,d,I,S_cap,S_ini)
%
% [J,S,u,w] = simulate_RS(u,d,I,S_cap,S_ini)

T = length(u);
d = d(:) ; % make sure d is a column vector
u = u(:) ; % make sure u is a column vector

w = nan(T,1);
S = nan(T+1,1); S(1) = S_ini;
for t = 1 : T
    u(t) = min(d(t),u(t)) ; % release must not exceed the demand
    u(t) = max(u(t),0)    ; % release must be >=0
    u(t) = min(u(t),S(t)+I(t) ); % release cannot exceed the available water volume
    S(t+1) = S(t) + I(t) - u(t);
    w(t)   = S(t+1) - min(S_cap,S(t+1));
    S(t+1) = S(t+1) - w(t);
end

J = sum((d - u).^2);
if nargout > 1
    varargout = {S(2:end), u, w};
end


