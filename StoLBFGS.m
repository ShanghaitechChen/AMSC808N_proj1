function [w,f,normgrad,time] = StoLBFGS(func,gfun,Y,w,Ng,Nh,itermax,M)
    %% Stochastic L-BFGS
    gam = 0.9; % line search step factor
    jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
    eta = 0.5; % backtracking stopping criterion factor

    tol = 1e-5;
    m = 5;
    dim = size(w,1);
    n = size(Y,1);
    Ng = min(n,Ng);
    Nh = min(n,Nh);
    s = zeros(dim,m); 
    y = zeros(dim,m);
    rho = zeros(1,m);

    f = zeros(itermax,1); 
    normgrad = zeros(itermax,1);
    time = zeros(itermax,1);
    
    % pair initialization
    g = gfun(1:n,Y,w);
    a = linesearch(1:n,Y,w,-g,g,func,eta,gam,jmax);
    w = w - a*g;
    g = gfun(1:n,Y,w);
    s(:,1) = w - w;
    y(:,1) = g - g;
    rho(1) = 1/(s(:,1)'*y(:,1));
    memory = 1;
    tic
    
    for iter = 1 : itermax
        if mod(iter,M) == 0
            batch_index = randperm(n,Nh);
        else
            batch_index = randperm(n,Ng);
        end
        g = gfun(batch_index,Y,w);
        if memory < m
            I = 1 : memory;
            p = finddirection(g,s(:,I),y(:,I),rho(I));
        else
            p = finddirection(g,s,y,rho);
        end
        [a,j] = linesearch(batch_index,Y,w,p,g,func,eta,gam,jmax);
        if j == jmax
            p = -g;
            a = linesearch(batch_index,Y,w,p,g,func,eta,gam,jmax);
        end

        if mod(iter,M) == 0
            step = a*p;
            w = w + step;
            g = gfun(batch_index,Y,w);
            s = circshift(s,[0,1]); 
            y = circshift(y,[0,1]);
            rho = circshift(rho,[0,1]);
            s(:,1) = step;
            y(:,1) = g - g;
            rho(1) = 1/(step'*y(:,1));
            memory = memory + 1;
        else
            step = a*p;
            w = w + step;
            g = gfun(batch_index,Y,w);
        end

        normgrad(iter) = norm(gfun(1:n,Y,w));
        f(iter) = func(1:n,Y,w);
        time(iter) = toc;

        if norm(g) < tol
            f(iter+1:end) = [];
            normgrad(iter+1:end) = [];
            fprintf('A local solution is found, iter = %d\n',iter);
            return
        end  
    end

    if iter == itermax
        fprintf('Stopped because the max number of iterations %d is performed\n',iter);
    end

end

%% line search algorithm
function [a,j] = linesearch(I,Y,w,p,g,func,eta,gam,jmax)
    a = 1;
    aux = eta*g'*p;
    for j = 0 : jmax
        xtry = w + a*p;
        f1 = func(I,Y,xtry);
        if f1 < func(I,Y,w) + a*aux
            break;
        else
            a = a*gam;
        end
    end
end