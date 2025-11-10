filepath='C:\Users\DELL EMC\Desktop\Machine_learning\our PDE\code\hyperbolic\different initials\for analysis-linear-test2\';
load([filepath 'training_data' num2str(1) '.mat'],'p_input','p_target','xL','xR','c1','c','N_m','N_t','T','N_tm','delta_t')

x=linspace(xL,xR,N_m+1);
xx=0.5*(x(1:N_m)+x(2:N_m+1));
h=(xR-xL)/N_m;

        %length of time interval

% N_t=floor(T/delta_t);
t=0:delta_t:T;


%%%%%%%%%%%%%%%%%%%%%%%%%%%
u_ini_ave=zeros(1,N_m);

for j=1:N_m
    u_ini_ave(j)=1/h*integral(@(x) u_exact_test(x,t(1)),x(j),x(j+1));
%       u_ini_ave(j)=1/h*integral(@(x) (1.*(x<=pi+t(1))+2.*(x>pi+t(1))),x(j),x(j+1));
end

u_ini_ave=double(u_ini_ave);

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
u_ave_col = u_ini_ave(:);

% 使用最终的索引矩阵一次性从 u_ave 中提取所有窗口数据
data_matrix = u_ave_col(circular_indices);

% 将结果矩阵的每一行（一个窗口）转换为一个 cell 中的列向量
u_initial = num2cell(data_matrix', 1);

% % 1. 先将 u_ave 转换为列向量
% u_ave_col = u_ini_ave(:);
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
% u_initial = num2cell(data_matrix', 1);
% 
% 



% u_initial{1}=[u_ini_ave(N_m-2) u_ini_ave(N_m-1) u_ini_ave(N_m) u_ini_ave(1) u_ini_ave(2)]'; 
% u_initial{2}=[u_ini_ave(N_m-1) u_ini_ave(N_m) u_ini_ave(1) u_ini_ave(2) u_ini_ave(3)]';
% u_initial{3}=[u_ini_ave(N_m) u_ini_ave(1) u_ini_ave(2) u_ini_ave(3) u_ini_ave(4)]';
% % u_initial{4}=[u_ini_ave(N_m-1) u_ini_ave(N_m) u_ini_ave(1) u_ini_ave(2) u_ini_ave(3) u_ini_ave(4) u_ini_ave(5) u_ini_ave(6) u_ini_ave(7)]';
% % u_initial{5}=[u_ini_ave(N_m) u_ini_ave(1) u_ini_ave(2) u_ini_ave(3) u_ini_ave(4) u_ini_ave(5) u_ini_ave(6) u_ini_ave(7) u_ini_ave(8)]';
% 
% u_initial{N_m-2}=[u_ini_ave(N_m-5) u_ini_ave(N_m-4) u_ini_ave(N_m-3) u_ini_ave(N_m-2) u_ini_ave(N_m-1)]';%periodic
% u_initial{N_m-1}=[u_ini_ave(N_m-4) u_ini_ave(N_m-3) u_ini_ave(N_m-2) u_ini_ave(N_m-1) u_ini_ave(N_m)]';%periodic
% u_initial{N_m}=[u_ini_ave(N_m-3) u_ini_ave(N_m-2) u_ini_ave(N_m-1) u_ini_ave(N_m) u_ini_ave(1)]';%periodic
% 
% for j=4:N_m-3
%     u_initial{j}=[u_ini_ave(j-3) u_ini_ave(j-2) u_ini_ave(j-1) u_ini_ave(j) u_ini_ave(j+1)]';
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%

load([filepath 'train_parameters.mat'],'W','b','M')

%standard nn part
zp=cell(N_m,N_t+1);
for j=1:N_m
    zp{j,1}=u_initial{j};
end

zx=zeros(N_m,N_t+1);

for i=1:N_m
zx(i,1)=zp{i,1}(c);
end
a=cell(M,1);

for i = 1:N_t
    %% 1. 向量化神经网络前向传播
    % 将cell数组 zp{:,i} 中的 N_m 个列向量合并成一个 D x N_m 的矩阵
    % 其中 D 是输入向量的维度 (这里是5)
    zp_matrix = [zp{:,i}]; % 或者使用 cat(2, zp{:,i})
    
    % --- 网络计算 (现在是矩阵运算) ---
    % 第一层
    a_matrix = transfer(W{1} * zp_matrix + b{1});
    
    % 隐藏层 (循环依然存在，但循环次数 M 通常远小于 N_m)
    for k = 2:M           
        a_matrix = transfer(W{k} * a_matrix + b{k});
    end
    
    % 输出层，包含残差连接
    % zp_matrix(c,:) 提取所有N_m个样本的第c个特征，形成一个行向量
    a_out_row = zp_matrix(c,:) + (W{M+1} * a_matrix + b{M+1});
    
    % 将计算结果（一个行向量）转置后存入 zx 的下一列
    zx(:, i+1) = a_out_row';
    
    z_col = zx(:, i+1);

%% 2. Vectorize the construction of the start_indices vector
% This vector will store the starting index in z_col for each window j.
start_indices = zeros(N_m, 1); % Pre-allocate for speed

% Rule for the first loop (j=1:c1-3)
j_range1 = 1:(c1-3);
start_indices(j_range1) = N_m - c1 + j_range1 - 1;

% Rule for the three special cases
start_indices(c1-2) = N_m - 3;
start_indices(c1-1) = N_m - 2;
start_indices(c1)   = N_m - 1;

% Rule for the second loop (j=c1+1:N_m)
j_range2 = (c1+1):N_m;
start_indices(j_range2) = j_range2 - c1;


%% 3. Build the full index matrix and handle circular boundaries
% Define the offsets for a 5-element window (0, 1, 2, 3, 4)
offsets = 0:c;

% Use broadcasting to create an N_m x 5 matrix of all "theoretical" indices
indices_matrix = start_indices + offsets;

% Apply the modulo operator to the whole matrix to handle circular wrap-around
% This maps all indices to the valid range [1, N_m]
circular_indices = mod(indices_matrix - 1, N_m) + 1;


%% 4. Extract data in one shot and format the output
% Use the final index matrix to grab all window data at once
data_matrix = z_col(circular_indices);

% Convert the resulting matrix into the final N_m x 1 cell array
% The two transposes ensure the output is a column of cells
zp(:, i+1) = num2cell(data_matrix', 1)';
%     %% 2. 向量化构造下一个时间步的输入 zp{:, i+1}
%     % 从 zx 中获取当前时间步的完整输出列
%     zx_current_col = zx(:, i+1);
%     
%     % 构造扩展向量以无缝处理循环边界
%     z_padded = [zx_current_col(end-(c-2):end); zx_current_col; zx_current_col(1)];
%     
%     % 使用广播机制生成所有滑动窗口的索引矩阵
%     indices = (1:N_m)' + (0:c); % 窗口大小为p+1+q=37
%     
%     % 一次性从扩展向量中提取所有窗口数据
%     data_matrix = z_padded(indices);
%     
%     % 将结果矩阵转换为cell数组，并存入zp的下一列
%     % data_matrix' -> (p+1+q)xN_m; num2cell(...,1) -> 1xN_m cell
%     % ...' -> N_mx1 cell，以匹配 zp(:, i+1) 的维度
%     zp(:, i+1) = num2cell(data_matrix', 1)';
end


% for i=1:N_t
%     for j=1:N_m
%         %run through the network after training is done
%         a{1}=transfer(W{1}*zp{j,i}(:,1)+b{1});
%         for k=2:M           
%             a{k}=transfer(W{k}*a{k-1}+b{k});
%         end
%         a_out=zp{j,i}(c,1)+(W{M+1}*a{M}+b{M+1});
%         
%         zx(j,i+1)=a_out;        
%     end
% 
%     
%   
%     for j=1:c1-3
%         zp{1,i+1}=[zx(N_m-c1+(j-1),i+1) zx(N_m-c1+(j),i+1) zx(N_m-c1+(j+1),i+1) zx(N_m-c1+(j+2),i+1) zx(N_m-c1+(j+3),i+1)]'; 
%     end
% 
%     zp{c1-2,i+1}=[zx(N_m-3,i+1) zx(N_m-2,i+1) zx(N_m-1,i+1) zx(N_m,i+1) zx(1,i+1)]';  
%     zp{c1-1,i+1}=[zx(N_m-2,i+1) zx(N_m-1,i+1) zx(N_m,i+1) zx(1,i+1) zx(2,i+1)]';
%     zp{c1,i+1}=[zx(N_m-1,i+1) zx(N_m,i+1) zx(1,i+1) zx(2,i+1) zx(3,i+1)]'; 
%     
%     for j=c1+1:N_m
%         zp{j,i+1}=[zx(j-c1,i+1) zx(j-c1+1,i+1) zx(j-c1+2,i+1) zx(j-c1+3,i+1) zx(j-c1+4,i+1)]';
%     end
% %     zp{N_m-2,i+1}=[zx(N_m-5,i+1) zx(N_m-4,i+1) zx(N_m-3,i+1) zx(N_m-2,i+1) zx(N_m-1,i+1)]';
% %     zp{N_m-1,i+1}=[zx(N_m-4,i+1) zx(N_m-3,i+1) zx(N_m-2,i+1) zx(N_m-1,i+1) zx(N_m,i+1)]';
% %     zp{N_m,i+1}=[zx(N_m-3,i+1) zx(N_m-2,i+1) zx(N_m-1,i+1) zx(N_m,i+1) zx(1,i+1)]'; 
% % 
% % end

u_ex=zeros(N_m,N_t+1);
u_ave=zeros(1,N_m);
for k=1:N_t+1
    for j=1:N_m
        u_ave(j)=1/h*integral(@(x) u_exact_test(x,t(k)),x(j),x(j+1));
    end
    u_ex(1:N_m,k)=double(u_ave);

end


u_ext_0=u_ex(:,1)'; 
uu_0=zx(1:N_m,1)';

u_ext_1=u_ex(:,5)'; 
uu_1=zx(1:N_m,5)';

% u_ext_2=u_ex(:,11)'; 
% uu_2=zx(1:N_m,11)';

u_ext_3=u_ex(:,end)'; 
uu_3=zx(1:N_m,end)';

figure(1)
plot(xx(1:end),uu_3(1:end),'m*','DisplayName','T3'); 
hold on
plot(xx(1:end),u_ext_3(1:end),'k-'); 
grid on

error_3=sqrt(h)*norm(uu_3-u_ext_3) 
error_3_inf=max(abs(uu_3-u_ext_3))

% figure(10) 
% plot(xx(1:end),uu_0(1:end),'r*','LineWidth',2,'Markersize',5); 
% hold on
% plot(xx(1:end),u_ext_0(1:end),'k-','LineWidth',2,'Markersize',5); 
% grid on
% axis([0,2*pi,-8,8]) 
% xlabel('x','FontSize',18) 
% ylabel('u','FontSize',18)
% legend('t=0','Exact','FontSize',12)
% 
% figure(11) 
% plot(xx(1:end),uu_1(1:end),'r*','LineWidth',2,'Markersize',5); 
% hold on
% plot(xx(1:end),u_ext_1(1:end),'k-','LineWidth',2,'Markersize',5); 
% grid on
% axis([0,2*pi,-8,8]) 
% xlabel('x','FontSize',18) 
% ylabel('u','FontSize',18)
% legend('t=\pi/5','Exact','FontSize',12)
% 
% figure(12) 
% plot(xx(1:end),uu_2(1:end),'r*','LineWidth',2,'Markersize',5); 
% hold on
% plot(xx(1:end),u_ext_2(1:end),'k-','LineWidth',2,'Markersize',5); 
% grid on
% axis([0,2*pi,-8,8]) 
% xlabel('x','FontSize',18) 
% ylabel('u','FontSize',18)
% legend('t=\pi/2','Exact','FontSize',12)
% 
% figure(13)
% plot(xx(1:end),uu_3(1:end),'r*','LineWidth',2,'Markersize',5); 
% hold on
% plot(xx(1:end),u_ext_3(1:end),'k-','LineWidth',2,'Markersize',5); 
% grid on
% axis([0,2*pi,-8,8]) 
% xlabel('x','FontSize',18) 
% ylabel('u','FontSize',18)
% legend('t=\pi','Exact','FontSize',12)
% 
% figure(9) 
% T0=plot(xx(1:end),uu_0(1:end),'b*','DisplayName','T0','LineWidth',2,'Markersize',5); 
% hold on
% plot(xx(1:end),u_ext_0(1:end),'k^-','LineWidth',2,'Markersize',5); 
% hold on
% T1=plot(xx(1:end),uu_1(1:end),'r*','DisplayName','T1','LineWidth',2,'Markersize',5); 
% hold on
% E=plot(xx(1:end),u_ext_1(1:end),'k^-','LineWidth',2,'Markersize',5); 
% hold on
% T2=plot(xx(1:end),uu_2(1:end),'g*','DisplayName','T2','LineWidth',2,'Markersize',5); 
% hold on
% plot(xx(1:end),u_ext_2(1:end),'k^-','LineWidth',2,'Markersize',5); 
% hold on
% T3=plot(xx(1:end),uu_3(1:end),'m*','DisplayName','T3','LineWidth',2,'Markersize',5); 
% hold on
% plot(xx(1:end),u_ext_3(1:end),'k^-','LineWidth',2,'Markersize',5); 
% grid on hold off
% axis([0,2*pi,-8,8]) 
% xlabel('x','FontSize',18) 
% ylabel('u','FontSize',18)
% legend([E,T0,T1,T2,T3],'Exact','t=0','t=0.025','t=0.05','t=0.1','FontSize',12)
% 
% 
% error_1=sqrt(h)*norm(uu_1-u_ext_1) 
% error_1_inf=max(abs(uu_1-u_ext_1))
% 
% error_2=sqrt(h)*norm(uu_2-u_ext_2) 
% error_2_inf=max(abs(uu_2-u_ext_2))



% figure(9)
% %%for delta t=1/80
% u_ext_0=u_ex(:,1)';
% uu_0=zx(1:N_m,1)';
% 
% u_ext_1=u_ex(:,3)';
% uu_1=zx(1:N_m,3)';
% 
% u_ext_2=u_ex(:,5)';
% uu_2=zx(1:N_m,5)';
% 
% u_ext_3=u_ex(:,end)';
% uu_3=zx(1:N_m,end)';
% 
% % 
% T0=plot(xx(1:end),uu_0(1:end),'b*','DisplayName','T0');
% hold on
% E=plot(xx(1:end),u_ext_0(1:end),'k^-');
% hold on
% T1=plot(xx(1:end),uu_1(1:end),'r*','DisplayName','T1');
% hold on
% plot(xx(1:end),u_ext_1(1:end),'k^-');
% hold on
% T2=plot(xx(1:end),uu_2(1:end),'g*','DisplayName','T2');
% hold on
% plot(xx(1:end),u_ext_2(1:end),'k^-');
% hold on
% T3=plot(xx(1:end),uu_3(1:end),'m*','DisplayName','T3');
% hold on
% plot(xx(1:end),u_ext_3(1:end),'k^-');
% grid on
% hold off
% axis([0,1,-0.2,1.2])
% xlabel('x','FontSize',18)
% ylabel('u','FontSize',18)
% legend([E,T0,T1,T2,T3],'Exact','t=0','t=0.025','t=0.05','t=0.1','FontSize',12)
% 
% 
% error_1=sqrt(h)*norm(uu_1-u_ext_1)
% error_1_inf=max(abs(uu_1-u_ext_1))
% 
% error_2=sqrt(h)*norm(uu_2-u_ext_2)
% error_2_inf=max(abs(uu_2-u_ext_2))
% 
% error_3=sqrt(h)*norm(uu_3-u_ext_3)
% error_3_inf=max(abs(uu_3-u_ext_3))



% figure(9)
% %%for delta t=1/160
% 
% u_ext_0=u_ex(:,1)';
% uu_0=zx(:,1)';
% 
% u_ext_1=u_ex(:,5)';
% uu_1=zx(:,5)';
% 
% u_ext_2=u_ex(:,9)';
% uu_2=zx(:,9)';
% 
% u_ext_3=u_ex(:,end)';
% uu_3=zx(:,end)';
% 
% % 
% T0=plot(xx(1:2:end),uu_0(1:2:end),'b*','DisplayName','T0');
% hold on
% E=plot(xx(1:2:end),u_ext_0(1:2:end),'k^-');
% hold on
% T1=plot(xx(1:2:end),uu_1(1:2:end),'r*','DisplayName','T1');
% hold on
% plot(xx(1:2:end),u_ext_1(1:2:end),'k^-');
% hold on
% T2=plot(xx(1:2:end),uu_2(1:2:end),'g*','DisplayName','T2');
% hold on
% plot(xx(1:2:end),u_ext_2(1:2:end),'k^-');
% hold on
% T3=plot(xx(1:2:end),uu_3(1:2:end),'m*','DisplayName','T3');
% hold on
% plot(xx(1:2:end),u_ext_3(1:2:end),'k^-');
% grid on
% hold off
% axis([0,1,-0.2,1.2])
% xlabel('x','FontSize',18)
% ylabel('u','FontSize',18)
% legend([E,T0,T1,T2,T3],'Exact','t=0','t=0.025','t=0.05','t=0.1','FontSize',12)
% 
% 
% error_1=sqrt(h)*norm(uu_1-u_ext_1)
% error_1_inf=max(abs(uu_1-u_ext_1))
% 
% error_2=sqrt(h)*norm(uu_2-u_ext_2)
% error_2_inf=max(abs(uu_2-u_ext_2))
% 
% error_3=sqrt(h)*norm(uu_3-u_ext_3)
% error_3_inf=max(abs(uu_3-u_ext_3))


% figure(9)
% %%for delta t=1/320
% 
% u_ext_0=u_ex(:,1)';
% uu_0=zx(:,1)';
% 
% u_ext_1=u_ex(:,9)';
% uu_1=zx(:,9)';
% 
% u_ext_2=u_ex(:,17)';
% uu_2=zx(:,17)';
% 
% u_ext_3=u_ex(:,end)';
% uu_3=zx(:,end)';
% 
% % 
% T0=plot(xx(1:2:end),uu_0(1:2:end),'b*','DisplayName','T0');
% hold on
% E=plot(xx(1:2:end),u_ext_0(1:2:end),'k^-');
% hold on
% T1=plot(xx(1:2:end),uu_1(1:2:end),'r*','DisplayName','T1');
% hold on
% plot(xx(1:2:end),u_ext_1(1:2:end),'k^-');
% hold on
% T2=plot(xx(1:2:end),uu_2(1:2:end),'g*','DisplayName','T2');
% hold on
% plot(xx(1:2:end),u_ext_2(1:2:end),'k^-');
% hold on
% T3=plot(xx(1:2:end),uu_3(1:2:end),'m*','DisplayName','T3');
% hold on
% plot(xx(1:2:end),u_ext_3(1:2:end),'k^-');
% grid on
% hold off
% axis([0,1,-0.2,1.2])
% xlabel('x','FontSize',18)
% ylabel('u','FontSize',18)
% legend([E,T0,T1,T2,T3],'Exact','t=0','t=0.025','t=0.05','t=0.1','FontSize',12)
% 
% 
% error_1=sqrt(h)*norm(uu_1-u_ext_1)
% error_1_inf=max(abs(uu_1-u_ext_1))
% 
% error_2=sqrt(h)*norm(uu_2-u_ext_2)
% error_2_inf=max(abs(uu_2-u_ext_2))
% 
% error_3=sqrt(h)*norm(uu_3-u_ext_3)
% error_3_inf=max(abs(uu_3-u_ext_3))

