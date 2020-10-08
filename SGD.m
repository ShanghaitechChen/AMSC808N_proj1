function [w, f, gnome, time] = SGD(fun, gfun, Y, w, batchsize, itermax, strategy, lr0)
%% Stochastic Gradient Descent 
% batch size: generally choose 64, 128, 256
% strategy: 1: fix lr 0.1; 2: lr = 0.1 * 1000/(1000+iter); 3: lr *= 0.999
% per iteration and decay *0.1 every 1000 iterations

    n = size(Y,1); % number of data
    batchsize = min(n,batchsize); % double check the batchsize
    lr = lr0;
    f = zeros(itermax,1); 
    gnome = zeros(itermax,1);
    time = zeros(itermax,1);
    tic

    for iter = 1 : itermax
        batch_index = randperm(n, batchsize);

        if strategy==2
            lr = lr0 * 1000/(1000+iter);
        else 
            if strategy==3
                if mod(iter, 10000)==0
                    lr = lr * 0.5;
                else
                    lr = lr * 0.999;
                end
            end
        end

        w = w - lr * gfun(batch_index,Y,w); % updating parameters
        f(iter) = fun(1:n,Y,w);
        gnome(iter) = norm(gfun(1:n,Y,w));
        time(iter) = toc;

        if gnome(iter) < 1e-10
            fprintf('A local solution is found, iter = %d\n',iter);
            return;
        end
    end
    
    if iter == itermax
        fprintf('Stopped because the max number of iterations %d is performed\n',iter);
    end
    
end