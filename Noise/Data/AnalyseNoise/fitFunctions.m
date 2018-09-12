close all;
clear all;
clc;


load('avrgSomaAutocorrData.mat')

F1 = @(p, t) exp(-t./p(1)).*cos((2*pi/p(2)).*t);
F2 = @(p, t) p(1).*exp(-t./p(2))-p(3).*exp(-t./p(4));


x0 = [400, 1000];
[p1,resnorm,~,exitflag,output] = lsqcurvefit(F1,x0,tt,XC_soma_avrg);

x0 = [1, 400, 1, 500];
[p2,resnorm2,~,exitflag2,output2] = lsqcurvefit(F2,x0,tt,XC_soma_avrg);

figure;
set(gcf,'units','points','position',[0,0,1200,500])
plot(tt, XC_soma_avrg)
hold on
plot(tt, F1(p1, tt))
hold on
plot(tt, F2(p2, tt))
title('UnBiased soma normed autocorrelation and fitted functions')
legend('Data', sprintf('exp(-t/%.2f)*cos(2*pi/%.2f*t)', p1(1), p1(2)), sprintf('%.2f*exp(-t/%.2f)-%.2f*exp(-t/%.2f)', p2(1), p2(2), p2(3), p2(4)))
xlabel('Time [ms]')
ylabel('Amp')
grid on
