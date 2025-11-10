clc;clear;
xL=0;    %computational domain
xR=2*pi;
% xR=10;

N_m=100;         %total number of subintervals

h=(xR-xL)/N_m;
x=linspace(xL,xR,N_m+1);
xx=0.5*(x(1:N_m)+x(2:N_m+1));

% T=pi^2/2;           %length of time interval

delta_t=10.987*pi*h;
% N_t=floor(T/delta_t);
N_t=10;
T=N_t*delta_t;

N_tm=1;

t=0:delta_t:T;

u_ave=zeros(1,N_m);
c1=38;
c=4;
for i=1:N_tm

    for j=1:N_m
        u_ave(j)=1/h*integral(@(x) u_exact(x,t(i)),x(j),x(j+1));
    end
u_ave=double(u_ave);
% 
% for j=1:c-3
%     p_input{j}=[u_ave(N_m-(c-j+1)) u_ave(N_m-(c-j)) u_ave(N_m-(c-j-1)) u_ave(N_m-(c-j-2)) u_ave(N_m-(c-j-3))]';
% end
% p_input{c-2}=[u_ave(N_m-3) u_ave(N_m-2) u_ave(N_m-1) u_ave(N_m) u_ave(1)]';
% p_input{c-1}=[u_ave(N_m-2) u_ave(N_m-1) u_ave(N_m) u_ave(1) u_ave(2)]';
% p_input{c}=[u_ave(N_m-1) u_ave(N_m) u_ave(1) u_ave(2) u_ave(3)]';
% 
% for j=c+1:N_m-3
%     p_input{j}=[u_ave(j-38) u_ave(j-37) u_ave(j-36) u_ave(j-35) u_ave(j-34)]';
% end
% 
% p_input{N_m-2}=[u_ave(N_m-2-38) u_ave(N_m-2-37) u_ave(N_m-2-36) u_ave(N_m-2-35) u_ave(N_m-2-34)]';
% p_input{N_m-1}=[u_ave(N_m-1-38) u_ave(N_m-1-37) u_ave(N_m-1-36) u_ave(N_m-1-35) u_ave(N_m-1-34)]';
% p_input{N_m}=[u_ave(N_m-38) u_ave(N_m-37) u_ave(N_m-36) u_ave(N_m-35) u_ave(N_m-34)]';
% 


% % 1. 为确保代码健壮，先将 u_ave 转换为列向量
% u_ave_col = u_ave(:);
% 
% % 2. 构造扩展向量以处理循环边界
% % 在头部拼接原向量的最后p个元素，在尾部拼接原向量的第q个元素
% % 这个填充的长度由窗口大小和偏移量决定
% u_padded = [u_ave_col(end-(c-2):end); u_ave_col; u_ave_col(1)];
% 
% % 3. 生成所有窗口的起始索引
% start_indices = (1:N_m)'; % 1, 2, ..., N_m 的列向量
% 
% % 4. 生成窗口内元素的偏移量 (窗口大小为p+1+q)
% offsets = 0:c; % 0, 1, 2, 3, 4 的行向量
% 
% % 5. 利用广播(broadcasting)机制创建完整的索引矩阵
% % 每一行对应 p_input{j} 需要的索引
% indices_matrix = start_indices + offsets;
% 
% % 6. 一次性从扩展向量中提取所有数据
% data_matrix = u_padded(indices_matrix);
% 
% % 7. 将结果矩阵的每一行转换为一个 cell, 并确保其为列向量
% % data_matrix' 将数据转置为 5xN_m
% % num2cell(..., 1) 将矩阵的每一列(5x1)放入一个cell中
% p_input = num2cell(data_matrix', 1);

%% 1. 向量化构建所有窗口的起始索引 (start_indices)
% 预分配一个列向量来存储 N_m 个起始索引
start_indices = zeros(N_m, 1);

% 规则 a) 处理 j = 1 到 c-3 的情况
j_range1 = 1:(c1-3);
start_indices(j_range1) = N_m - (c1 - j_range1 + 1);

% 规则 b) 处理 j = c-2, c-1, c 的三个特殊点
start_indices(c1-2) = N_m - 3;
start_indices(c1-1) = N_m - 2;
start_indices(c1)   = N_m - 1;

% 规则 c) 处理 j = c+1 到 N_m 的情况
j_range2 = (c1+1):N_m;
start_indices(j_range2) = j_range2 - c1;

%% 2. 构建完整的 5-element 窗口索引矩阵
% 创建窗口内元素的偏移量 (0, 1, 2, 3, 4)
offsets = 0:c;

% 利用广播机制，将每个起始索引扩展成一个完整的窗口索引行
% 得到一个 N_m x 5 的“理论”索引矩阵
indices_matrix = start_indices + offsets;

% 使用 mod 函数处理循环边界，将理论索引转换为 1 到 N_m 范围内的有效索引
circular_indices = mod(indices_matrix - 1, N_m) + 1;

%% 3. 一次性提取所有数据并转换为 cell 数组
% 确保 u_ave 是列向量
u_ave_col = u_ave(:);

% 使用最终的索引矩阵一次性从 u_ave 中提取所有窗口数据
data_matrix = u_ave_col(circular_indices);

% 将结果矩阵的每一行（一个窗口）转换为一个 cell 中的列向量
p_input = num2cell(data_matrix', 1);



%generate Target data
for j=1:N_m
    u_ave(j)=1/h*integral(@(x) u_exact(x,t(i+1)),x(j),x(j+1));
end
u_ave=double(u_ave);
p_target=u_ave';

filename=strcat('training_data',num2str(i),'.mat');
save(filename,'p_input','p_target','xL','xR','c1','c','N_m','N_t','T','N_tm','delta_t');

set(figure(i), 'visible', 'on');
yy1=zeros(1,N_m);
for k=1:N_m
    yy1(k)=p_input{k}(4);
end
% yy1(N_m+1)=p_input{N_m}(2);

yy2=p_target';

plot(xx,yy1,'r*')
hold on
plot(xx,yy2,'b^')
hold off
grid on
xlabel('x')
ylabel('u')
end