function u=u_exact(x,t)
% u=3*sin(5*(x-t))+4*cos(7*(x-t));
% u=1+2*(x-t)+(x-t).^3+3*(x-t).^5+4*(x-t).^8;
u=sin(2*(x-t));

% u=1+2*(x-t)+(x-t).^3;
% if x-t<=pi
%     u=1;
% else
%     u=2;
% end
% end

