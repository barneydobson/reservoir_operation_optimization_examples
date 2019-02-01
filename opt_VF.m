function H = opt_VF( I, d, g_end, discr_s, discr_u, s_cap )

T     = length(I) ;
n_s  = length(discr_s);
sys_param.discr_s = discr_s ;
sys_param.discr_u = discr_u ;
sys_param.S_cap = s_cap     ;
H     = zeros( n_s , T+1 ) ; % create Bellman VF
H(:,end) = g_end ; % initialize Bellman VF to penalty function
for t = T:-1:1    
   for i = 1 : length( sys_param.discr_s ) ;
      sys_param.d = d(t) ;
      H(i,t) = solve_VF( H(:,t+1), sys_param.discr_s(i), I(t) , sys_param ) ;
   end
end
