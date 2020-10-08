function Project1main()
close all
%% read data
A2012 = readmatrix('A2012.csv');
A2016 = readmatrix('A2016.csv');
% Format for A2012 and A2016:
% FIPS, County, #DEM, #GOP, then <str> up to Unemployment Rate
str = ["Median Income", "Migration Rate", "Birth Rate",...
"Death Rate", "Bachelor Rate", "Unemployment Rate","log(#Votes)"];
%
% remove column county that is read by matlab as NaN
A2012(:,2) = [];
A2016(:,2) = [];
%% Remove rows with missing data
A = A2016;
% remove all rows with missing data
ind = find(~isfinite(A(:,2)) |  ~isfinite(A(:,3)) | ~isfinite(A(:,4)) ...
    | ~isfinite(A(:,5)) | ~isfinite(A(:,6)) | ~isfinite(A(:,7)) ...
    | ~isfinite(A(:,8)) | ~isfinite(A(:,9)));
A(ind,:) = [];
%% select CA, OR, WA, NJ, NY counties (uncomment for Q1, comment for Q2,3,4)
ind = find((A(:,1)>=6000 & A(:,1)<=6999) ...  %CA
 | (A(:,1)>=53000 & A(:,1)<=53999) ...        %WA
 | (A(:,1)>=34000 & A(:,1)<=34999) ...        %NJ  
 | (A(:,1)>=36000 & A(:,1)<=36999) ...        %NY
 | (A(:,1)>=41000 & A(:,1)<=41999));          %OR
A = A(ind,:);
[n,dim] = size(A);

%% assign labels: -1 = dem, 1 = GOP
idem = find(A(:,2) >= A(:,3));
igop = find(A(:,2) < A(:,3));
num = A(:,2)+A(:,3);
label = zeros(n,1);
label(idem) = -1;
label(igop) = 1;

%% select max subset of data with equal numbers of dem and gop counties (comment for Q1, uncomment for Q2,3,4)
% ngop = length(igop);
% ndem = length(idem);
% if ngop > ndem
%     rgop = randperm(ngop,ndem);
%     Adem = A(idem,:);
%     Agop = A(igop(rgop),:);
%     A = [Adem;Agop];
% else
%     rdem = randperm(ndem,ngop);
%     Agop = A(igop,:);
%     Adem = A(idem(rdem),:);
%     A = [Adem;Agop];
% end  
% [n,dim] = size(A)
% idem = find(A(:,2) >= A(:,3));
% igop = find(A(:,2) < A(:,3));
% num = A(:,2)+A(:,3);
% label = zeros(n,1);
% label(idem) = -1;
% label(igop) = 1;

%% set up data matrix and visualize
close all
figure;
hold on; grid;
X = [A(:,4:9),log(num)];
X(:,1) = X(:,1)/1e4;
% select three data types that distinguish dem and gop counties the most
i1 = 1; % Median Income
i2 = 7; % log(# votes)
i3 = 5; % Bachelor Rate
plot3(X(idem,i1),X(idem,i2),X(idem,i3),'.','color','b','Markersize',20);
plot3(X(igop,i1),X(igop,i2),X(igop,i3),'.','color','r','Markersize',20);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(str(i1),'Fontsize',fsz);
ylabel(str(i2),'Fontsize',fsz);
zlabel(str(i3),'Fontsize',fsz);
%% rescale data to [0,1] and visualize
figure;
hold on; grid;
XX = X(:,[i1,i2,i3]); % data matrix
% rescale all data to [0,1]
xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
X1 = (XX(:,1)-xmin)/(xmax-xmin);
X2 = (XX(:,2)-ymin)/(ymax-ymin);
X3 = (XX(:,3)-zmin)/(zmax-zmin);
XX = [X1,X2,X3];
plot3(XX(idem,1),XX(idem,2),XX(idem,3),'.','color','b','Markersize',20);
plot3(XX(igop,1),XX(igop,2),XX(igop,3),'.','color','r','Markersize',20);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(str(i1),'Fontsize',fsz);
ylabel(str(i2),'Fontsize',fsz);
zlabel(str(i3),'Fontsize',fsz);
%% set up optimization problem
[n,dim] = size(XX);
lam = 0.01;
Y = (label*ones(1,dim + 1)).*[XX,ones(n,1)];
w = [-1;-1;1;1];
fun = @(I,Y,w)fun0(I,Y,w,lam);
gfun = @(I,Y,w)gfun0(I,Y,w,lam);
Hvec = @(I,Y,w,v)Hvec0(I,Y,w,v,lam);

%% Soft SVM, Q1
e = ones(n,1);
w = SINewton(fun,gfun,Hvec,Y,w,64,100);
xi0 = max(e-Y*w,0);
w = [w; xi0];
c = 1e3;
gfun = @(x)SVM_gradient(x,dim,n,c);
Hfun = @(x)SVM_hessian(dim,n);
A = [Y, eye(n); zeros(n,dim+1), eye(n)];
b = [ones(n, 1); zeros(n, 1)];
W = find(A*w - b == 0);
% call ASM.n
w = ASM(w,gfun,Hfun,A,b,W);
% fetch w and b
w = w(1:4);

%% SGD, Q2
% function [w, f, gnome] = SGD(fun, gfun, Y, w, batchsize, itermax, strategy, lr0)
% batch size: generally choose 64, 128, 256
% strategy: 1: fix lr 0.1; 2: lr = 0.1 * 1000/(1000+iter); 3: lr *= 0.999 per iteration and decay *0.1 every 10000 iterations
% iter_max = 10000;
% batch_size = 256; % batch size
% lam = 0.01; % the parameter \lambda
% fun = @(I,Y,w)fun0(I,Y,w,lam);
% gfun = @(I,Y,w)gfun0(I,Y,w,lam);
% f_stat = zeros(iter_max, 1000); % record statistics for function value 
% g_stat = f_stat; % record statistics for grad norm
% t_stat = f_stat; % record statistics for running time
% for i=1:1000
%     w = [-1;-1;1;1];
%     [w,f,gnorm,time] = SGD(fun, gfun, Y, w, batch_size, iter_max, 1, 0.1);
%     f_stat(:,i) = f;
%     g_stat(:,i) = gnorm;
%     t_stat(:,i) = time;
% end
% f = mean(f_stat, 2);
% gnorm = mean(g_stat, 2);
% time = mean(t_stat, 2);

%% SINewton, Q3
% iter_max = 1000; % max iteration num
% batch_size = 256; % batch size
% f_stat = zeros(iter_max+1, 1000); % record statistics for function value 
% g_stat = zeros(iter_max, 1000); % record statistics for grad norm
% t_stat = f_stat; % record statistics for running time
% for i=1:1000
%     w = [-1;-1;1;1];
%     [w,f,gnorm,time] = SINewton(fun,gfun,Hvec,Y,w,batch_size,iter_max);
%     f_stat(:,i) = f;
%     g_stat(:,i) = gnorm;
%     t_stat(:,i) = time;
% end
% f = mean(f_stat, 2);
% gnorm = mean(g_stat, 2);
% time = mean(t_stat, 2);

%% LBFGS, Q4
% Ng = 128; % batch size for gradient
% Nh = 256; % batch size for Hessian
% M=50; % update frequency for the pairs
% iter_max = 500; % max iteration num
% f_stat = zeros(iter_max, 100); % record statistics for function value 
% g_stat = zeros(iter_max, 100); % record statistics for grad norm
% t_stat = f_stat; % record statistics for running time
% for i=1:100
%     w = [-1;-1;1;1];
%     [w,f,gnorm,time] = StoLBFGS(fun,gfun,Y,w,Ng,Nh,iter_max,M);
%     f_stat(:,i) = f;
%     g_stat(:,i) = gnorm;
%     t_stat(:,i) = time;
% end
% f = mean(f_stat, 2);
% gnorm = mean(g_stat, 2);
% time = mean(t_stat, 2);

%% Plotting
fprintf('w = [%d,%d,%d], b = %d\n',w(1),w(2),w(3),w(4));

xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
nn = 50;
[xx,yy,zz] = meshgrid(linspace(xmin,xmax,nn),linspace(ymin,ymax,nn),...
    linspace(zmin,zmax,nn));
plane = w(1)*xx+w(2)*yy+w(3)*zz+w(4);
p = patch(isosurface(xx,yy,zz,plane,0));
p.FaceColor = 'green';
p.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);

%% ploting function value, grad norm vs iteration and running time (uncomment for Q2-4)
% figure;
% hold on;
% grid;
% niter = length(f);
% plot((0:niter-1)',f,'Linewidth',2);
% set(gca,'Fontsize',fsz);
% xlabel('k','Fontsize',fsz);
% ylabel('f','Fontsize',fsz);
% 
% %%
% figure;
% hold on;
% grid;
% niter = length(f);
% plot(time(1:niter),f,'Linewidth',2);
% set(gca,'Fontsize',fsz);
% xlabel('t','Fontsize',fsz);
% ylabel('f','Fontsize',fsz);
% 
% %%
% figure;
% hold on;
% grid;
% niter = length(gnorm);
% plot((0:niter-1)',gnorm,'Linewidth',2);
% set(gca,'Fontsize',fsz);
% set(gca,'YScale','log');
% xlabel('k','Fontsize',fsz);
% ylabel('|| stoch grad f||','Fontsize',fsz);

end

%%
function f = fun0(I,Y,w,lam)
f = sum(log(1 + exp(-Y(I,:)*w)))/length(I) + 0.5*lam*w'*w;
end
%%
function g = gfun0(I,Y,w,lam)
aux = exp(-Y(I,:)*w);
d1 = size(Y,2);
g = sum(-Y(I,:).*((aux./(1 + aux))*ones(1,d1)),1)'/length(I) + lam*w;
end
%%
function Hv = Hvec0(I,Y,w,v,lam)
aux = exp(-Y(I,:)*w);
d1 = size(Y,2);
Hv = sum(Y(I,:).*((aux.*(Y(I,:)*v)./((1+aux).^2)).*ones(1,d1)),1)' + lam*v;
end

%% Soft-SVM gradient, hessian and contraint
% grad for soft SVM
function g_svm = SVM_gradient(x,d,n,c) 
    H = [eye(d), zeros(d,n+1); zeros(n+1,d), zeros(n+1,n+1)];
    v = [zeros(d+1, 1); ones(n, 1)]; 
    g_svm = H * x + c * v;
end
% Hessian for soft SVM
function H_svm = SVM_hessian(d,n) 
    H_svm = [eye(d), zeros(d,n+1); zeros(n+1,d), zeros(n+1,n+1)];
end