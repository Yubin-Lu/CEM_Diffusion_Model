close all
T=10;
factors=5;
facerr=[];
dti_rec=[];
nX0=5;
X0=rand(nX0,2);
nmc=10000;
idx_rec=zeros(length(factors),nX0);
N_samples = 10^5;
rnd = randn(nmc,2);
% X0=[1,2;1,0;4,5];
figure
t=(linspace(0,1,100)).^(factors)*T;
Nt=length(t)-1;
ifac=1;
d=size(X0,2);
err_rec=[];
tic
Xrec=zeros(nmc,d);
for imc=1:nmc
    X=rnd(imc,:);
    for i = 1:Nt
        ti=t(Nt-i+2);
        dti=t(Nt-i+2)-t(Nt-i+1);
        alp=exp(-ti/2);
        X=X+sqrt(dti)*randn(1,d);
        Y0=(X-X0*alp)/sqrt(1-alp^2);
        wY0=exp(-sum(Y0.*Y0/2,2));
        wY0=wY0./sum(wY0);
        drift=-X/2+1/sqrt(1-alp^2)*sum(Y0.*wY0,1);
        X=X-drift*dti;
        if(i==30)
            Xt1(imc,:) = X;
        elseif(i==60)
            Xt2(imc,:) = X;
        elseif(i==80)
            Xt3(imc,:) = X;
        elseif(i==90)
            Xt4(imc,:) = X;
        end
    end
    Xrec(imc,:)=X;
    [err_rec(imc),idx]=min(sqrt(sum((X-X0).^2,2)));
    idx_rec(ifac,idx)=idx_rec(ifac,idx)+1;
end
toc
% histogram2(Xrec(:,1), Xrec(:,2))
% histogram(Xrec(:,1))
freq = idx_rec/nmc
plot(idx_rec,'r-*')
ylim([0,2500])
xlabel('Points')
ylabel('Frequency')

figure
subplot(2,3,1)
scatter(rnd(:,1),rnd(:,2),'r')
% xlabel('x')
ylabel('y')
xlim([-5,5])
ylim([-5,5])
subplot(2,3,2)
scatter(Xt1(:,1),Xt1(:,2),'r')
% xlabel('x')
% ylabel('y')
xlim([-5,5])
ylim([-5,5])
subplot(2,3,3)
scatter(Xt2(:,1),Xt2(:,2),'r')
% xlabel('x')
% ylabel('y')
xlim([-5,5])
ylim([-5,5])
subplot(2,3,4)
scatter(Xt3(:,1),Xt3(:,2),'r')
xlabel('x')
ylabel('y')
xlim([-2,2])
ylim([-2,2])
subplot(2,3,5)
scatter(Xt4(:,1),Xt4(:,2),'r')
xlabel('x')
% ylabel('y')
xlim([-1,1])
ylim([-1,1])
subplot(2,3,6)
% plot(1)
scatter(Xrec(:,1),Xrec(:,2),'r')
xlabel('x')
% ylabel('y')
xlim([-1,1])
ylim([-1,1])

save analytic_multiPoints freq rnd Xt1 Xt2 Xt3 Xt4 Xrec