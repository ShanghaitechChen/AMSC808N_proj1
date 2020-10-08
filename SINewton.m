function [w,f,normgrad,time] = SINewton(fun,gfun,Hvec,Y,w,batchsize,iter_max)
    rho = 0.1;
    gam = 0.9;
    jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
    eta = 0.5;
    CGimax = 20; % max number of CG iterations
    n = size(Y,1);
    bsz = min(n,batchsize); % batch size
    [n,~] = size(Y);
    I = 1:n;
    f = zeros(iter_max + 1,1);
    time = f;
    f(1) = fun(I,Y,w);
    normgrad = zeros(iter_max,1);
    time(1) = 0;
    nfail = 0;
    nfailmax = 5*ceil(n/bsz);
    tic
    for k = 1 : iter_max
        Ig = randperm(n,bsz);
        IH = randperm(n,bsz);
        Mvec = @(v)Hvec(IH,Y,w,v);
        b = gfun(Ig,Y,w);
        normgrad(k) = norm(b);
        s = CG(Mvec,-b,-b,CGimax,rho);
        a = 1;
        f0 = fun(Ig,Y,w);
        aux = eta*b'*s;
        for j = 0 : jmax
            wtry = w + a*s;
            f1 = fun(Ig,Y,wtry);
            if f1 < f0 + a*aux
                fprintf('Linesearch: j = %d, f1 = %d, f0 = %d\n',j,f1,f0);
                break;
            else
                a = a*gam;
            end
        end
        if j < jmax
            w = wtry;
        else
            nfail = nfail + 1;
        end
        f(k + 1) = fun(I,Y,w);
        time(k+1) = toc;
        fprintf('k = %d, a = %d, f = %d\n',k,a,f(k+1));
        if nfail > nfailmax
            f(k+2:end) = [];
            normgrad(k+1:end) = [];
            break;
        end
    end
end
        
        
    
    
