function [J,varargout] = simulate_VF(H,d,I,S_cap,S_ini,discr_u,discr_s)
%
% [J,S,u,w] = simulate_VF(H,d,I,S_cap,S_ini,discr_u,discr_s)

T = length(I);
d = d(:) ; % make sure d is a column vector
u = zeros(T,1) ; % preallocate releases

sys_param.discr_s = discr_s ;
sys_param.discr_u = discr_u ;
sys_param.S_cap = S_cap     ;

w = nan(T,1);
S = nan(T+1,1); S(1) = S_ini;
for t = 1 : T
    sys_param.d = d(t) ;
    [~,idx_u] = solve_VF( H(:,t+1), S(t), I(t) , sys_param ) ;
    u(t) = discr_u(idx_u(end));
    
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


