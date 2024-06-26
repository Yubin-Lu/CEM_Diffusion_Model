close all
clc; clear
T=10;
factors=linspace(1,5,5);
facerr=[];
dti_rec=[];
nX0=20;
X0=rand(nX0,2);
nmc=10000;
idx_rec=zeros(length(factors),nX0);
% X0=[1,2;1,0;4,5];
figure
for ifac=1:length(factors)
    t=(linspace(0,1,100)).^(factors(ifac))*T;
    Nt=length(t)-1;

    XT=[0.5,0.5];
    d=size(X0,2);
    err_rec=[];
    tic
    Xrec=zeros(nmc,d);
    for imc=1:nmc
        X=XT;
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
        end
        [err_rec(imc),idx]=min(sqrt(sum((X-X0).^2,2)));
        idx_rec(ifac,idx)=idx_rec(ifac,idx)+1;
        Xrec(imc,:)=X;
    end
    toc
    dti_rec(ifac)=t(2)-t(1);
    facerr(ifac)=mean(abs(err_rec));
    subplot(2,3,ifac)
    scatter(Xrec(:,1),Xrec(:,2))
    hold on
    scatter(X0(:,1),X0(:,2),'+')
    legend('generated','training data')
end
subplot(2,3,6)
loglog(dti_rec, facerr)
facerr./sqrt(dti_rec)
legs={};
figure()
for ifac=1:length(factors)
    plot(sort(idx_rec(ifac,:)))
    legs{ifac}=[num2str(factors(ifac))];
    hold on
end
legend(legs)
