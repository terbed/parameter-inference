close all;
clear all;
clc;


load('SomaAutocorr_notnormed.mat')


F0 = @(p, t) p(1).*p(2).*exp(-p(2).*t);
F1 = @(p, t) p(1).*exp(-t./p(2)).*cos((2*pi/p(3)).*t);
F2 = @(p, t) p(1).*exp(-t./p(2))-p(3).*exp(-t./p(4));

% tt = tt(1:50000);
% XC_soma_avrg = XC_soma_avrg(1:50000);

x0 = [0.11, 300, 870];
[p1,resnorm,~,exitflag,output] = lsqcurvefit(F1,x0,tt,XC_soma_avrg);

x00 = [1,400, 1,500];
[p2,resnorm2,~,exitflag2,output2] = lsqcurvefit(F2,x00,tt,XC_soma_avrg);

x000 = [10, 100];
[p0,resnorm0,~,exitflag0,output0] = lsqcurvefit(F0, x000,tt,XC_soma_avrg);

disp("p(2)*p(1).*exp(-p(2)*t)");
p0
resnorm0

disp(" p(1)*exp(-t./p(2)).*cos((2*pi/p(3)).*t);");
p1
resnorm

disp("p(1).*exp(-t./p(2))-p(3).*exp(-t./p(4))");
p2
resnorm2


figure;
set(gcf,'units','points','position',[0,0,1200,500])
plot(tt, XC_soma_avrg, 'LineWidth', 2)
hold on
plot(tt, F1(p1, tt), 'LineWidth', 2)
hold on
plot(tt, F2(p2, tt), '--', 'LineWidth', 2)
hold on
plot(tt, F0(p0, tt), ':', 'LineWidth', 2)
title('UnBiased soma not-normed autocorrelation and fitted functions')
lgd = legend('Data', sprintf('%.4f*exp(-t/%.2f)*cos(2*pi/%.2f*t)', p1(1), p1(2), p1(3)), sprintf('%.2f*exp(-t/%.2f)-%.2f*exp(-t/%.2f)', p2(1), p2(2), p2(3), p2(4)), sprintf('%.4f*%.2f*exp(-%.4f*t)', p0(2), p0(1), p0(2)));
lgd.FontSize = 20;
xlabel('Time [ms]')
ylabel('Amp')
grid on

figure;
set(gcf,'units','points','position',[0,0,1200,500])
plot(tt, XC_soma_avrg,'k', 'LineWidth', 2)
hold on
plot(tt, F1(x0, tt), 'r--', 'LineWidth', 2)
grid on
plot(tt, XC_soma)
lgd = legend('Data', sprintf('%.4f*exp(-t/%.2f)*cos(2*pi/%.2f*t)', x0(1), x0(2), x0(3)));
lgd.FontSize = 16;
title('UnBiased soma not-normed autocorrelation and fitted functions by hand')


% -----------------------------------RESULTS for the normed autocorr fit
% pp(1).*exp(-p(2)*t)
% 
% p0 =
% 
%     1.1538    0.0167
% 
% 
% resnorm0 =
% 
%   766.2340
% -------------------------------------------------------------------------
%  exp(-t./p(1)).*cos((2*pi/p(2)).*t);
% 
% p1 =
% 
%   215.2253  758.7082
% 
% 
% resnorm =
% 
%   305.0915
% ------------------------------------------------------------------------
% p(1).*exp(-t./p(2))-p(3).*exp(-t./p(4))
% 
% p2 =
% 
%    10.4973  156.6380    9.4165  174.5898
% 
% 
% resnorm2 =
% 
%   171.0141


% ------------------------------------- RESULTS FOT THE NOT NORMED AUT. FIT
% --------------------------------------------------------------------------
% 
% p(2)*p(1).*exp(-p(2)*t)
% 
% p0 =
% 
%     8.7546    0.0156
% 
% 
% resnorm0 =
% 
%    13.7027
% -----------------------------------------------------
%  p(1)*exp(-t./p(2)).*cos((2*pi/p(3)).*t);
% 
% p1 =
% 
%     0.0947  314.1812  822.5770
% resnorm =
% 
%     3.2424
% ---------------------------------------------------------
% p(1).*exp(-t./p(2))-p(3).*exp(-t./p(4))
% 
% p2 =
% 
%     1.2515  188.5518    1.1303  210.8669
% 
% 
% resnorm2 =
% 
%     4.0052