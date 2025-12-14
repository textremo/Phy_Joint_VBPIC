clear;
clc;
% this script is to calculate the threshold of the system
% parameters
pmax_beg = 6;
pmax = 10; % the maximal path number
SNR_d = 14;
No = 10^(-SNR_d/10);
powXd = 1;
% probabilties
pNo = 0.9999;
pPath = 0.9999;
pilot_num = 4;

thres = chi2inv(pNo, 2*pilot_num)/2/pilot_num;
disp("Not data + noise threshold")
%% methods 1
disp("----method1----")
% disp("1 pilot case: Show the threshold when we can confirm it is a path!");
% for p = pmax_beg:pmax
%     SINR_p_linear = (chi2inv(pNo, 2)/chi2inv((1-pPath), 2)-1)*p*(powXd+No); % the SINR of pilot (pilot/(Xd + z))
%     SINR_p = 10*log10(SINR_p_linear);
%     powXp = SINR_p_linear*(powXd + No);
%     SNR_p = 10*log10(powXp/No);
%     fprintf("Path is %d: SNR_p is %.8f, SNR_p-SNR_d is %.8f\n", p, SNR_p, (SNR_p-SNR_d));
% end

disp(pilot_num + " pilot case: Show the threshold when we can confirm it is a path!");
for p = pmax_beg:pmax
    SINR_p_linear = (chi2inv(pNo, 2*pilot_num)/chi2inv((1-pPath)*(powXd+No), 2*pilot_num)-1)*p; % the SINR of pilot (pilot/(Xd + z))
    SINR_p = 10*log10(SINR_p_linear);
    powXp = SINR_p_linear*(powXd + No);
    SNR_p = 10*log10(powXp/No);
    fprintf("Path is %d: SNR_p is %.8f, SNR_p-SNR_d is %.8f\n", p, SNR_p, (SNR_p-SNR_d));
end

%% method 2
disp("----method2----")

disp(pilot_num + " pilot case: Show the threshold when we can confirm it is a path!");
for p = pmax_beg:pmax
    SINR_p_linear = (chi2inv(pNo, 2)/chi2inv((1-pPath), 2)-1)*p*(powXd+No)/pilot_num; % the SINR of pilot (pilot/(Xd + z))
    SINR_p = 10*log10(SINR_p_linear);
    powXp = SINR_p_linear*(powXd + No);
    SNR_p = 10*log10(powXp/No);
    fprintf("Path is %d: SNR_p is %.8f, SNR_p-SNR_d is %.8f\n", p, SNR_p, (SNR_p-SNR_d));
end

% A = randi(100, 4, 1) + 1i * randi(100, 4, 1);% 随机生成一个4*1的列向量A
% B = null(A.');% 求解A的转置的零空间
% b1 = B(:, 1); b2 = B(:, 2); b3 = B(:, 3);% 得到A的零空间的标准正交基
% A_orth = randi(100)*b1 + randi(100)*b2 + randi(100)*b3;% 随机构建A的正交向量
% A_orth = conj(A_orth);