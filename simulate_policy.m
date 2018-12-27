function [obj, varargout] = simulate_policy(parameters,demand,flows,storage_cap,storage_initial)
T = length(flows);
releases = zeros(T,1);
spills = zeros(T,1);
storage = zeros(T,1);
for t = 1 : T
    if t == 1
        releases(t) = parameters(1) + storage_initial*parameters(2) + flows(t)*parameters(3);
    else
        releases(t) = parameters(1) + storage(t-1)*parameters(2) + flows(t)*parameters(3);
    end
    releases(t) = min(demand(t),releases(t));
    releases(t) = max(releases(t),0);
    if t == 1
        releases(t) = min(releases(t),storage_initial + flows(t));
        storage(t) = storage_initial + flows(t) - releases(t);
    else
        releases(t) = min(releases(t),storage(t-1) + flows(t));
        storage(t) = storage(t-1) + flows(t) - releases(t);
    end
    spills(t) = storage(t) - min(storage_cap,storage(t));
    storage(t) = storage(t) - spills(t);
end
obj = sum((demand - releases).^2);
if nargout > 1
    varargout = {storage, releases, spills};
end